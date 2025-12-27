use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use num_bigint::BigUint;
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
use openvm_te_transpiler::Rv32EdwardsOpcode;

use super::TeAddExecutor;
use crate::edwards_chip::curves::{get_curve_type, te_add, CurveType};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct TeAddPreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> TeAddExecutor<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut TeAddPreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.offset);

        // Pre-compute flag_idx
        let needs_setup = self.expr.needs_setup();
        let mut flag_idx = self.expr.num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.opcode_flag_idx.len() {
                    flag_idx = self.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = TeAddPreCompute {
            expr: &self.expr,
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32EdwardsOpcode::SETUP_TE_ADD as usize;

        Ok(is_setup)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $pre_compute:ident, $is_setup:ident) => {
        if let Some(curve_type) = {
            let modulus = &$pre_compute.expr.builder.prime;
            let a_coeff = &$pre_compute.expr.setup_values[0];
            let d_coeff = &$pre_compute.expr.setup_values[1];
            get_curve_type(modulus, a_coeff, d_coeff)
        } {
            match ($is_setup, curve_type) {
                (true, CurveType::ED25519) => Ok($execute_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::ED25519 as u8 },
                    true,
                >),
                (false, CurveType::ED25519) => Ok($execute_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::ED25519 as u8 },
                    false,
                >),
            }
        } else if $is_setup {
            Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }, true>)
        } else {
            Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }, false>)
        }
    };
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InterpreterExecutor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<TeAddPreCompute>()
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
        let pre_compute: &mut TeAddPreCompute = data.borrow_mut();

        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
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
        let pre_compute: &mut TeAddPreCompute = data.borrow_mut();

        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> AotExecutor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InterpreterMeteredExecutor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<TeAddPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<TeAddPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<TeAddPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> AotMeteredExecutor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &TeAddPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for both points
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| exec_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    if IS_SETUP {
        let input_prime = BigUint::from_bytes_le(read_data[0][..BLOCKS / 2].as_flattened());
        let input_a = BigUint::from_bytes_le(read_data[0][BLOCKS / 2..].as_flattened());
        let input_d = BigUint::from_bytes_le(read_data[1][..BLOCKS / 2].as_flattened());

        if input_prime != pre_compute.expr.prime {
            let err = ExecutionError::Fail {
                pc,
                msg: "TeAdd: mismatched prime",
            };
            return Err(err);
        }

        if input_a != pre_compute.expr.setup_values[0] {
            let err = ExecutionError::Fail {
                pc,
                msg: "TeAdd: mismatched a",
            };
            return Err(err);
        }

        if input_d != pre_compute.expr.setup_values[1] {
            let err = ExecutionError::Fail {
                pc,
                msg: "TeAdd: mismatched d",
            };
            return Err(err);
        }
    }

    let output_data = if CURVE_TYPE == u8::MAX || IS_SETUP {
        let read_data: DynArray<u8> = read_data.into();
        run_field_expression_precomputed::<true>(
            pre_compute.expr,
            pre_compute.flag_idx as usize,
            &read_data.0,
        )
        .into()
    } else {
        te_add::<CURVE_TYPE, BLOCKS, BLOCK_SIZE>(read_data)
    };

    let rd_val = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Write output data to memory
    for (i, block) in output_data.into_iter().enumerate() {
        exec_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &TeAddPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<TeAddPreCompute>()).borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let e2_pre_compute: &E2PreCompute<TeAddPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<TeAddPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, 1);
    let pre_compute = &e2_pre_compute.data;
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(pre_compute, exec_state)
}
