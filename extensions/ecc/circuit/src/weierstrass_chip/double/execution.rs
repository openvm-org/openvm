use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use num_bigint::BigUint;
use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_mod_circuit_builder::{run_field_expression_precomputed, FieldExpr};
use openvm_stark_backend::p3_field::PrimeField32;

use super::EcDoubleExecutor;
use crate::weierstrass_chip::curves::{ec_double, get_curve_type, CurveType};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcDoublePreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 1],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> EcDoubleExecutor<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcDoublePreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
        let Instruction {
            opcode, a, b, d, e, ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
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

        let rs_addrs = [b as u8];
        *data = EcDoublePreCompute {
            expr: &self.expr,
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize;

        Ok(is_setup)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident,$pre_compute:ident,$is_setup:ident) => {
        if let Some(curve_type) = {
            let modulus = &$pre_compute.expr.builder.prime;
            let a_coeff = &$pre_compute.expr.setup_values[0];
            get_curve_type(modulus, a_coeff)
        } {
            match ($is_setup, curve_type) {
                (true, CurveType::K256) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }, true>)
                }
                (true, CurveType::P256) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::P256 as u8 }, true>)
                }
                (true, CurveType::BN254) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BN254 as u8 }, true>)
                }
                (true, CurveType::BLS12_381) => Ok($execute_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BLS12_381 as u8 },
                    true,
                >),
                (false, CurveType::K256) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }, false>)
                }
                (false, CurveType::P256) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::P256 as u8 }, false>)
                }
                (false, CurveType::BN254) => {
                    Ok($execute_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BN254 as u8 }, false>)
                }
                (false, CurveType::BLS12_381) => Ok($execute_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BLS12_381 as u8 },
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

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> Executor<F>
    for EcDoubleExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcDoublePreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EcDoublePreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_impl, pre_compute, is_setup)
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
        let pre_compute: &mut EcDoublePreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_tco_handler, pre_compute, is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> MeteredExecutor<F>
    for EcDoubleExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcDoublePreCompute>>()
    }

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
        let pre_compute: &mut E2PreCompute<EcDoublePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;

        dispatch!(execute_e2_impl, pre_compute_pure, is_setup)
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
        let pre_compute: &mut E2PreCompute<EcDoublePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;

        dispatch!(execute_e2_tco_handler, pre_compute_pure, is_setup)
    }
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &EcDoublePreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for the point
    let read_data: [[u8; BLOCK_SIZE]; BLOCKS] = {
        let address = rs_vals[0];
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    };

    if IS_SETUP {
        let input_prime = BigUint::from_bytes_le(read_data[..BLOCKS / 2].as_flattened());

        if input_prime != pre_compute.expr.builder.prime {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "EcDouble: mismatched prime",
            });
            return;
        }

        // Extract second field element as the a coefficient
        let input_a = BigUint::from_bytes_le(read_data[BLOCKS / 2..].as_flattened());
        let coeff_a = &pre_compute.expr.setup_values[0];
        if input_a != *coeff_a {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "EcDouble: mismatched coeff_a",
            });
            return;
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
        ec_double::<CURVE_TYPE, BLOCKS, BLOCK_SIZE>(read_data)
    };

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Write output data to memory
    for (i, block) in output_data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &EcDoublePreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let e2_pre_compute: &E2PreCompute<EcDoublePreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(
        &e2_pre_compute.data,
        vm_state,
    );
}
