use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_AS, LocalOpcode,
};
use openvm_riscv_transpiler::BranchEqualOpcode;
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
        if d.as_canonical_u32() != RV64_REGISTER_AS {
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

macro_rules! dispatch {
    ($execute_impl:ident, $is_bne:ident) => {
        if $is_bne {
            Ok($execute_impl::<_, true>)
        } else {
            Ok($execute_impl::<_, false>)
        }
    };
}

impl<F, A, const NUM_LIMBS: usize> InterpreterExecutor<F> for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqualPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
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
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
    }
}

impl<F, A, const NUM_LIMBS: usize> InterpreterMeteredExecutor<F>
    for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchEqualPreCompute>>()
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
        let data: &mut E2PreCompute<BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
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
        let data: &mut E2PreCompute<BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
    }
}
#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &BranchEqualPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let mut pc = exec_state.pc();
    let rs1 = exec_state.vm_read_bytes::<8>(RV64_REGISTER_AS, pre_compute.a as u32);
    let rs2 = exec_state.vm_read_bytes::<8>(RV64_REGISTER_AS, pre_compute.b as u32);
    if (rs1 == rs2) ^ IS_NE {
        pc = (pc as isize + pre_compute.imm) as u32;
    } else {
        pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BranchEqualPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BranchEqualPreCompute>()).borrow();
    execute_e12_impl::<CTX, IS_NE>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchEqualPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BranchEqualPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, IS_NE>(&pre_compute.data, exec_state);
}
