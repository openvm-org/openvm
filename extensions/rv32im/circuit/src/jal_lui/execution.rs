use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        E2PreCompute, ExecuteFunc, ExecutionCtxTrait, Executor, MeteredExecutionCtxTrait,
        MeteredExecutor, StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, JAL};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{get_signed_imm, Rv32JalLuiExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalLuiPreCompute {
    signed_imm: i32,
    a: u8,
}

impl<A> Rv32JalLuiExecutor<A> {
    /// Return (IS_JAL, ENABLED)
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut JalLuiPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let local_opcode = Rv32JalLuiOpcode::from_usize(
            inst.opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET),
        );
        let is_jal = local_opcode == JAL;
        let signed_imm = get_signed_imm(is_jal, inst.c);

        *data = JalLuiPreCompute {
            signed_imm,
            a: inst.a.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok((is_jal, enabled))
    }
}

impl<F, A> Executor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalLuiPreCompute>()
    }

    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(inst, data)?;
        let fn_ptr = match (is_jal, enabled) {
            (true, true) => execute_e1_impl::<_, _, true, true>,
            (true, false) => execute_e1_impl::<_, _, true, false>,
            (false, true) => execute_e1_impl::<_, _, false, true>,
            (false, false) => execute_e1_impl::<_, _, false, false>,
        };
        Ok(fn_ptr)
    }
}

impl<F, A> MeteredExecutor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalLuiPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(inst, &mut data.data)?;
        let fn_ptr = match (is_jal, enabled) {
            (true, true) => execute_e2_impl::<_, _, true, true>,
            (true, false) => execute_e2_impl::<_, _, true, false>,
            (false, true) => execute_e2_impl::<_, _, false, true>,
            (false, false) => execute_e2_impl::<_, _, false, false>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: &JalLuiPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let JalLuiPreCompute { a, signed_imm } = *pre_compute;

    let rd = if IS_JAL {
        let rd_data = (vm_state.pc + DEFAULT_PC_STEP).to_le_bytes();
        let next_pc = vm_state.pc as i32 + signed_imm;
        debug_assert!(next_pc >= 0);
        vm_state.pc = next_pc as u32;
        rd_data
    } else {
        let imm = signed_imm as u32;
        let rd = imm << 12;
        vm_state.pc += DEFAULT_PC_STEP;
        rd.to_le_bytes()
    };

    if ENABLED {
        vm_state.vm_write(RV32_REGISTER_AS, a as u32, &rd);
    }

    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalLuiPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalLuiPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(&pre_compute.data, vm_state);
}
