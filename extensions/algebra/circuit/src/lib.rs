use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use derive_more::derive::{Deref, DerefMut};
use openvm_circuit::{
    arch::{
        execution::ExecuteFunc,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        DynArray, E2PreCompute, ExecutionError, InsExecutorE1, InsExecutorE2, Result,
        VmSegmentState,
    },
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpr, FieldExpressionStep,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterStep;
use openvm_stark_backend::p3_field::PrimeField32;

use self::fields::{
    field_operation, fp2_operation, get_field_type_from_modulus, FieldType, Operation,
};

pub mod fp2_chip;
pub mod modular_chip;

mod fp2;
pub use fp2::*;
mod modular_extension;
pub use modular_extension::*;
mod fp2_extension;
pub use fp2_extension::*;
mod config;
pub use config::*;
pub mod fields;

#[derive(Clone, InstructionExecutor, Deref, DerefMut)]
pub struct FieldExprVecHeapStep<const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>(
    FieldExpressionStep<Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
);

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldExpressionPreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    FieldExprVecHeapStep<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldExpressionPreCompute<'a>,
    ) -> Result<(bool, Operation)> {
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
            return Err(ExecutionError::InvalidInstruction(pc));
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

        let op = match local_opcode {
            0 => Operation::Add,
            1 => Operation::Sub,
            2 => Operation::Mul,
            3 => Operation::Div,
            _ => return Ok((needs_setup, Operation::Add)), // Setup operations
        };

        Ok((needs_setup, op))
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    InsExecutorE1<F> for FieldExprVecHeapStep<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FieldExpressionPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let pre_compute: &mut FieldExpressionPreCompute = data.borrow_mut();

        let (is_setup, op) = self.pre_compute_impl(pc, inst, pre_compute)?;

        if is_setup {
            Ok(execute_e1_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(field_type) = {
            let modulus = &pre_compute.expr.prime;
            get_field_type_from_modulus(modulus)
        } {
            match (field_type, op) {
                (FieldType::BN254, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BN254, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BN254, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BN254, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::BLS12_381, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BLS12_381, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BLS12_381, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BLS12_381, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::K256, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::K256, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::K256, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::K256, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::P256, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::P256, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::P256, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::P256, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Div as u8 },
                >),
            }
        } else {
            Ok(execute_e1_generic_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>)
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    InsExecutorE2<F> for FieldExprVecHeapStep<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FieldExpressionPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let pre_compute: &mut E2PreCompute<FieldExpressionPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (is_setup, op) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        if is_setup {
            Ok(execute_e2_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(field_type) = {
            let modulus = &pre_compute.data.expr.prime;
            get_field_type_from_modulus(modulus)
        } {
            match (field_type, op) {
                (FieldType::BN254, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BN254, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BN254, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BN254, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BN254 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::BLS12_381, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BLS12_381, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BLS12_381, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BLS12_381, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::K256, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::K256, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::K256, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::K256, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::K256 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::P256, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::P256, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::P256, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::P256, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    IS_FP2,
                    { FieldType::P256 as u8 },
                    { Operation::Div as u8 },
                >),
            }
        } else {
            Ok(execute_e2_generic_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2>)
        }
    }
}

unsafe fn execute_e1_setup_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>(pre_compute, vm_state);
}

unsafe fn execute_e2_setup_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>(&pre_compute.data, vm_state);
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2, FIELD_TYPE, OP>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, IS_FP2, FIELD_TYPE, OP>(
        &pre_compute.data,
        vm_state,
    );
}

unsafe fn execute_e1_generic_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute = pre_compute.borrow();
    execute_e12_generic_impl::<_, _, BLOCKS, BLOCK_SIZE>(pre_compute, vm_state);
}

unsafe fn execute_e2_generic_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_generic_impl::<_, _, BLOCKS, BLOCK_SIZE>(&pre_compute.data, vm_state);
}

unsafe fn execute_e12_setup_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &FieldExpressionPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });
    let read_data: DynArray<u8> = read_data.into();

    let writes = run_field_expression_precomputed::<true>(
        pre_compute.expr,
        pre_compute.flag_idx as usize,
        &read_data.0,
    );

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    let data: [[u8; BLOCK_SIZE]; BLOCKS] = writes.into();
    for (i, block) in data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &FieldExpressionPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    let output_data = if IS_FP2 {
        fp2_operation::<FIELD_TYPE, BLOCKS, BLOCK_SIZE, OP>(read_data)
    } else {
        field_operation::<FIELD_TYPE, BLOCKS, BLOCK_SIZE, OP>(read_data)
    };

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    for (i, block) in output_data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

unsafe fn execute_e12_generic_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &FieldExpressionPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });
    let read_data: DynArray<u8> = read_data.into();

    let writes = run_field_expression_precomputed::<false>(
        pre_compute.expr,
        pre_compute.flag_idx as usize,
        &read_data.0,
    );

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    let data: [[u8; BLOCK_SIZE]; BLOCKS] = writes.into();
    for (i, block) in data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}
