use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC},
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::Rv64JalrExecutor;
use crate::adapters::{rv64_address_add_imm, rv64_bytes_to_u32};
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalrPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<A> Rv64JalrExecutor<A> {
    /// Return true if enabled.
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut JalrPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        if inst.d.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = JalrPreCompute {
            imm_extended,
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok(enabled)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $enabled:ident) => {
        if $enabled {
            Ok($execute_impl::<_, true>)
        } else {
            Ok($execute_impl::<_, false>)
        }
    };
}

impl<F, A> InterpreterExecutor<F> for Rv64JalrExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalrPreCompute>()
    }
    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for Rv64JalrExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalrPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &JalrPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pc = exec_state.pc();
    let rs1 =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1 = rv64_bytes_to_u32(rs1);
    let unaligned_to_pc = rv64_address_add_imm(rs1, pre_compute.imm_extended);
    // JALR clears bit 0 before jumping.
    let to_pc = unaligned_to_pc & !1;
    debug_assert!(to_pc <= u64::from(MAX_ALLOWED_PC));
    let to_pc = to_pc as u32;
    let mut rd = [0u8; RV64_REGISTER_NUM_LIMBS];
    rd[..4].copy_from_slice(&(pc + DEFAULT_PC_STEP).to_le_bytes());

    if ENABLED {
        exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.a as u32, &rd);
    }

    exec_state.set_pc(to_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &JalrPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<JalrPreCompute>()).borrow();
    execute_e12_impl::<CTX, ENABLED>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalrPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JalrPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, ENABLED>(&pre_compute.data, exec_state);
}
