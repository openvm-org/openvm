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
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::BranchEqualExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchEqualPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize> BranchEqualExecutor<A, NUM_LIMBS> {
    /// Return `is_bne`, true if the local opcode is BNE.
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BranchEqualPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BranchEqualPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(local_opcode == BranchEqualOpcode::BNE)
    }
}

impl<F, A, const NUM_LIMBS: usize> Executor<F> for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqualPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        let fn_ptr = if is_bne {
            execute_e1_impl::<_, _, true>
        } else {
            execute_e1_impl::<_, _, false>
        };
        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_LIMBS: usize> MeteredExecutor<F> for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchEqualPreCompute>>()
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
        let data: &mut E2PreCompute<BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let fn_ptr = if is_bne {
            execute_e2_impl::<_, _, true>
        } else {
            execute_e2_impl::<_, _, false>
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &BranchEqualPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    if (rs1 == rs2) ^ IS_NE {
        vm_state.pc = (vm_state.pc as isize + pre_compute.imm) as u32;
    } else {
        vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    }
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BranchEqualPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_NE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchEqualPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_NE>(&pre_compute.data, vm_state);
}
