use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use num_bigint::BigUint;
use openvm_algebra_transpiler::{Fp2Opcode, Rv32ModularArithmeticOpcode};
use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_mod_circuit_builder::{run_field_expression_precomputed, FieldExpr};
use openvm_stark_backend::p3_field::PrimeField32;

use super::FieldExprVecHeapExecutor;
use crate::fields::{
    field_operation, fp2_operation, get_field_type, get_fp2_field_type, FieldType, Operation,
};

macro_rules! generate_field_dispatch {
    (
        $field_type:expr,
        $op:expr,
        $blocks:expr,
        $block_size:expr,
        $execute_fn:ident,
        [$(($curve:ident, $operation:ident)),* $(,)?]
    ) => {
        match ($field_type, $op) {
            $(
                (FieldType::$curve, Operation::$operation) => Ok($execute_fn::<
                    _,
                    _,
                    $blocks,
                    $block_size,
                    false,
                    { FieldType::$curve as u8 },
                    { Operation::$operation as u8 },
                >),
            )*
        }
    };
}

macro_rules! generate_fp2_dispatch {
    (
        $field_type:expr,
        $op:expr,
        $blocks:expr,
        $block_size:expr,
        $execute_fn:ident,
        [$(($curve:ident, $operation:ident)),* $(,)?]
    ) => {
        match ($field_type, $op) {
            $(
                (FieldType::$curve, Operation::$operation) => Ok($execute_fn::<
                    _,
                    _,
                    $blocks,
                    $block_size,
                    true,
                    { FieldType::$curve as u8 },
                    { Operation::$operation as u8 },
                >),
            )*
            _ => panic!("Unsupported fp2 field")
        }
    };
}

macro_rules! dispatch {
    ($execute_impl:ident,$execute_generic_impl:ident,$execute_setup_impl:ident,$pre_compute:ident,$op:ident) => {
        if let Some(op) = $op {
            let modulus = &$pre_compute.expr.prime;
            if IS_FP2 {
                if let Some(field_type) = get_fp2_field_type(modulus) {
                    generate_fp2_dispatch!(
                        field_type,
                        op,
                        BLOCKS,
                        BLOCK_SIZE,
                        $execute_impl,
                        [
                            (BN254Coordinate, Add),
                            (BN254Coordinate, Sub),
                            (BN254Coordinate, Mul),
                            (BN254Coordinate, Div),
                            (BLS12_381Coordinate, Add),
                            (BLS12_381Coordinate, Sub),
                            (BLS12_381Coordinate, Mul),
                            (BLS12_381Coordinate, Div),
                        ]
                    )
                } else {
                    Ok($execute_generic_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>)
                }
            } else if let Some(field_type) = get_field_type(modulus) {
                generate_field_dispatch!(
                    field_type,
                    op,
                    BLOCKS,
                    BLOCK_SIZE,
                    $execute_impl,
                    [
                        (K256Coordinate, Add),
                        (K256Coordinate, Sub),
                        (K256Coordinate, Mul),
                        (K256Coordinate, Div),
                        (K256Scalar, Add),
                        (K256Scalar, Sub),
                        (K256Scalar, Mul),
                        (K256Scalar, Div),
                        (P256Coordinate, Add),
                        (P256Coordinate, Sub),
                        (P256Coordinate, Mul),
                        (P256Coordinate, Div),
                        (P256Scalar, Add),
                        (P256Scalar, Sub),
                        (P256Scalar, Mul),
                        (P256Scalar, Div),
                        (BN254Coordinate, Add),
                        (BN254Coordinate, Sub),
                        (BN254Coordinate, Mul),
                        (BN254Coordinate, Div),
                        (BN254Scalar, Add),
                        (BN254Scalar, Sub),
                        (BN254Scalar, Mul),
                        (BN254Scalar, Div),
                        (BLS12_381Coordinate, Add),
                        (BLS12_381Coordinate, Sub),
                        (BLS12_381Coordinate, Mul),
                        (BLS12_381Coordinate, Div),
                        (BLS12_381Scalar, Add),
                        (BLS12_381Scalar, Sub),
                        (BLS12_381Scalar, Mul),
                        (BLS12_381Scalar, Div),
                    ]
                )
            } else {
                Ok($execute_generic_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>)
            }
        } else {
            Ok($execute_setup_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>)
        }
    };
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldExpressionPreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldExpressionPreCompute<'a>,
    ) -> Result<Option<Operation>, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.0.offset);

        let needs_setup = self.0.expr.needs_setup();
        let mut flag_idx = self.0.expr.num_flags() as u8;
        if needs_setup {
            if let Some(opcode_position) = self
                .0
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                if opcode_position < self.0.opcode_flag_idx.len() {
                    flag_idx = self.0.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = FieldExpressionPreCompute {
            a: a as u8,
            rs_addrs,
            expr: &self.0.expr,
            flag_idx,
        };

        if IS_FP2 {
            let is_setup = local_opcode == Fp2Opcode::SETUP_ADDSUB as usize
                || local_opcode == Fp2Opcode::SETUP_MULDIV as usize;

            let op = if is_setup {
                None
            } else {
                match local_opcode {
                    x if x == Fp2Opcode::ADD as usize => Some(Operation::Add),
                    x if x == Fp2Opcode::SUB as usize => Some(Operation::Sub),
                    x if x == Fp2Opcode::MUL as usize => Some(Operation::Mul),
                    x if x == Fp2Opcode::DIV as usize => Some(Operation::Div),
                    _ => unreachable!(),
                }
            };

            Ok(op)
        } else {
            let is_setup = local_opcode == Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize
                || local_opcode == Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize;

            let op = if is_setup {
                None
            } else {
                match local_opcode {
                    x if x == Rv32ModularArithmeticOpcode::ADD as usize => Some(Operation::Add),
                    x if x == Rv32ModularArithmeticOpcode::SUB as usize => Some(Operation::Sub),
                    x if x == Rv32ModularArithmeticOpcode::MUL as usize => Some(Operation::Mul),
                    x if x == Rv32ModularArithmeticOpcode::DIV as usize => Some(Operation::Div),
                    _ => unreachable!(),
                }
            };

            Ok(op)
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool> Executor<F>
    for FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FieldExpressionPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut FieldExpressionPreCompute = data.borrow_mut();
        let op = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(
            execute_e1_handler,
            execute_e1_generic_handler,
            execute_e1_setup_handler,
            pre_compute,
            op
        )
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut FieldExpressionPreCompute = data.borrow_mut();
        let op = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(
            execute_e1_handler,
            execute_e1_generic_handler,
            execute_e1_setup_handler,
            pre_compute,
            op
        )
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    MeteredExecutor<F> for FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FieldExpressionPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<FieldExpressionPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let op = self.pre_compute_impl(pc, inst, pre_compute_pure)?;

        dispatch!(
            execute_e2_handler,
            execute_e2_generic_handler,
            execute_e2_setup_handler,
            pre_compute_pure,
            op
        )
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<FieldExpressionPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let op = self.pre_compute_impl(pc, inst, pre_compute_pure)?;

        dispatch!(
            execute_e2_handler,
            execute_e2_generic_handler,
            execute_e2_setup_handler,
            pre_compute_pure,
            op
        )
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &FieldExpressionPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| exec_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    let output_data = if IS_FP2 {
        fp2_operation::<FIELD_TYPE, BLOCKS, BLOCK_SIZE, OP>(read_data)
    } else {
        field_operation::<FIELD_TYPE, BLOCKS, BLOCK_SIZE, OP>(read_data)
    };

    let rd_val = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    for (i, block) in output_data.into_iter().enumerate() {
        exec_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_generic_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &FieldExpressionPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| exec_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });
    let read_data_dyn: DynArray<u8> = read_data.into();

    let writes = run_field_expression_precomputed::<true>(
        pre_compute.expr,
        pre_compute.flag_idx as usize,
        &read_data_dyn.0,
    );

    let rd_val = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    let data: [[u8; BLOCK_SIZE]; BLOCKS] = writes.into();
    for (i, block) in data.into_iter().enumerate() {
        exec_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_setup_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &FieldExpressionPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    // Read the first input (which should be the prime)
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, addr as u32)));
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| exec_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    // Extract first field element as the prime
    let input_prime = if IS_FP2 {
        BigUint::from_bytes_le(read_data[0][..BLOCKS / 2].as_flattened())
    } else {
        BigUint::from_bytes_le(read_data[0].as_flattened())
    };

    if input_prime != pre_compute.expr.prime {
        let err = ExecutionError::Fail {
            pc: *pc,
            msg: "ModularSetup: mismatched prime",
        };
        return Err(err);
    }

    let read_data_dyn: DynArray<u8> = read_data.into();

    let writes = run_field_expression_precomputed::<true>(
        pre_compute.expr,
        pre_compute.flag_idx as usize,
        &read_data_dyn.0,
    );

    let rd_val = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    let data: [[u8; BLOCK_SIZE]; BLOCKS] = writes.into();
    for (i, block) in data.into_iter().enumerate() {
        exec_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_setup_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>(pre_compute, instret, pc, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_setup_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>(
        &pre_compute.data,
        instret,
        pc,
        exec_state,
    )
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2, FIELD_TYPE, OP>(
        pre_compute,
        instret,
        pc,
        exec_state,
    );
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2, FIELD_TYPE, OP>(
        &pre_compute.data,
        instret,
        pc,
        exec_state,
    );
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_generic_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_generic_impl::<_, _, BLOCKS, BLOCK_SIZE>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_generic_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_generic_impl::<_, _, BLOCKS, BLOCK_SIZE>(
        &pre_compute.data,
        instret,
        pc,
        exec_state,
    );
}
