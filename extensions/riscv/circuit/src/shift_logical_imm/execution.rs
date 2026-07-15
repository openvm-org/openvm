use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::ShiftLogicalImmExecutor;
use crate::adapters::imm_to_rv64_u64;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct ShiftLogicalImmPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftLogicalImmExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftLogicalImmPreCompute,
    ) -> Result<ShiftImmOpcode, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let c = c.as_canonical_u32();
        if d.as_canonical_u32() != RV64_REGISTER_AS
            || e.as_canonical_u32() != RV64_IMM_AS
            || c >= (NUM_LIMBS * LIMB_BITS) as u32
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftLogicalImmPreCompute {
            c: imm_to_rv64_u64(c),
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(ShiftImmOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftLogicalImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftLogicalImmPreCompute>()
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
        let data: &mut ShiftLogicalImmPreCompute = data.borrow_mut();
        let opcode = self.pre_compute_impl(pc, inst, data)?;
        Ok(match opcode {
            ShiftImmOpcode::SLLI => execute_e1_handler::<F, Ctx, true, SllOp>,
            ShiftImmOpcode::SRLI => execute_e1_handler::<F, Ctx, true, SrlOp>,
            ShiftImmOpcode::SRAI => unreachable!(),
        })
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
        let data: &mut ShiftLogicalImmPreCompute = data.borrow_mut();
        let opcode = self.pre_compute_impl(pc, inst, data)?;
        Ok(match opcode {
            ShiftImmOpcode::SLLI => execute_e1_handler::<F, Ctx, true, SllOp>,
            ShiftImmOpcode::SRLI => execute_e1_handler::<F, Ctx, true, SrlOp>,
            ShiftImmOpcode::SRAI => unreachable!(),
        })
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftLogicalImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftLogicalImmPreCompute>>()
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
        let data: &mut E2PreCompute<ShiftLogicalImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(match opcode {
            ShiftImmOpcode::SLLI => execute_e2_handler::<F, Ctx, true, SllOp>,
            ShiftImmOpcode::SRLI => execute_e2_handler::<F, Ctx, true, SrlOp>,
            ShiftImmOpcode::SRAI => unreachable!(),
        })
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
        let data: &mut E2PreCompute<ShiftLogicalImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(match opcode {
            ShiftImmOpcode::SLLI => execute_e2_handler::<F, Ctx, true, SllOp>,
            ShiftImmOpcode::SRLI => execute_e2_handler::<F, Ctx, true, SrlOp>,
            ShiftImmOpcode::SRAI => unreachable!(),
        })
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: ImmOp>(
    pre_compute: &ShiftLogicalImmPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1 = u64::from_le_bytes(rs1);
    let rs2 = pre_compute.c;
    let rd = <OP as ImmOp>::compute(rs1, rs2);
    exec_state.vm_write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        RV64_REGISTER_AS,
        pre_compute.a as u32,
        &rd.to_le_bytes(),
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ImmOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftLogicalImmPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftLogicalImmPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ImmOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftLogicalImmPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<ShiftLogicalImmPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait ImmOp {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct SllOp;
impl ImmOp for SllOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_shl((rs2 & 63) as u32)
    }
}
struct SrlOp;
impl ImmOp for SrlOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_shr((rs2 & 63) as u32)
    }
}
