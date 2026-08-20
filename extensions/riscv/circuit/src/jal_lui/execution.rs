use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, riscv::REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::JalLuiOpcode::{self, JAL};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{get_signed_imm, run_jal_lui, JalLuiExecutor};
use crate::adapters::byte_ptr_to_u16_ptr_value;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalLuiPreCompute {
    signed_imm: i32,
    a: u8,
}

impl JalLuiExecutor {
    /// Return (IS_JAL, ENABLED)
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut JalLuiPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let local_opcode =
            JalLuiOpcode::from_usize(inst.opcode.local_opcode_idx(JalLuiOpcode::CLASS_OFFSET));
        let is_jal = local_opcode == JAL;
        let signed_imm =
            get_signed_imm(is_jal, inst.c).ok_or(StaticProgramError::InvalidInstruction(pc))?;

        *data = JalLuiPreCompute {
            signed_imm,
            a: inst.a.as_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok((is_jal, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_jal:ident, $enabled:ident) => {
        match ($is_jal, $enabled) {
            (true, true) => Ok($execute_impl::<_, true, true>),
            (true, false) => Ok($execute_impl::<_, true, false>),
            (false, true) => Ok($execute_impl::<_, false, true>),
            (false, false) => Ok($execute_impl::<_, false, false>),
        }
    };
}

impl<F> InterpreterExecutor<F> for JalLuiExecutor
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            JalLuiOpcode::from_usize(opcode - JalLuiOpcode::CLASS_OFFSET)
        )
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalLuiPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
    }
}

impl<F> InterpreterMeteredExecutor<F> for JalLuiExecutor
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalLuiPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_JAL: bool, const ENABLED: bool>(
    pre_compute: &JalLuiPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let JalLuiPreCompute { a, signed_imm } = *pre_compute;
    let (pc, rd) = run_jal_lui(IS_JAL, exec_state.pc(), signed_imm);

    if ENABLED {
        exec_state.vm_write(REGISTER_AS, byte_ptr_to_u16_ptr_value(a as u32), &rd);
    } else {
        exec_state.ctx.advance_timestamp(1);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_JAL: bool, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &JalLuiPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<JalLuiPreCompute>()).borrow();
    execute_e12_impl::<CTX, IS_JAL, ENABLED>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    CTX: MeteredExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalLuiPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JalLuiPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, IS_JAL, ENABLED>(&pre_compute.data, exec_state);
}
