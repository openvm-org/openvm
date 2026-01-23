use openvm_circuit::arch::{
    E1ExecutionCtx, E2ExecutionCtx, E2PreCompute, ExecuteFunc, InsExecutorE1, InsExecutorE2,
    StaticProgramError, VmSegmentState,
};
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS};
use openvm_stark_backend::p3_field::PrimeField32;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use crate::auipc_dazl::chip::Rv32AuipcDazlStep;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct AuiPcPreCompute {
    imm: u32,
    a: u8,
}

impl<F> InsExecutorE1<F> for Rv32AuipcDazlStep
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AuiPcPreCompute>()
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(|pre_compute, vm_state| {
            let pre_compute: &AuiPcPreCompute = pre_compute.borrow();
            unsafe {
                execute_e1_impl(pre_compute, vm_state);
            }
        })
    }
}

#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &AuiPcPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rd = crate::run_auipc(vm_state.pc, pre_compute.imm);
    vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

impl<F> InsExecutorE2<F> for Rv32AuipcDazlStep
where
    F: PrimeField32,
{
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AuiPcPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<AuiPcPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(|pre_compute, vm_state| {
            let pre_compute: &E2PreCompute<AuiPcPreCompute> = pre_compute.borrow();
            vm_state
                .ctx
                .on_height_change(pre_compute.chip_idx as usize, 1);
            unsafe {
                execute_e1_impl(&pre_compute.data, vm_state);
            }
        })
    }
}

impl Rv32AuipcDazlStep {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AuiPcPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, c: imm, d, .. } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let imm = imm.as_canonical_u32();
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        *data = AuiPcPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

// returns rd_data
#[inline(always)]
pub fn run_auipc(pc: u32, imm: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    rd.to_le_bytes()
}
