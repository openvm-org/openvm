use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::MulHOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::MulHExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MulHPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut MulHPreCompute,
    ) -> Result<MulHOpcode, StaticProgramError> {
        *data = MulHPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(MulHOpcode::from_usize(
            inst.opcode.local_opcode_idx(MulHOpcode::CLASS_OFFSET),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            MulHOpcode::MULH => Ok($execute_impl::<_, _, MulHOp>),
            MulHOpcode::MULHSU => Ok($execute_impl::<_, _, MulHSuOp>),
            MulHOpcode::MULHU => Ok($execute_impl::<_, _, MulHUOp>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> Executor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<MulHPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut MulHPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(inst, pre_compute)?;
        dispatch!(execute_e1_impl, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut MulHPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(inst, pre_compute)?;
        dispatch!(execute_e1_tco_handler, local_opcode)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MulHPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<MulHPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_impl, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<MulHPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_tco_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: &MulHPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd = <OP as MulHOperation>::compute(rs1, rs2);
    vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MulHPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MulHPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, vm_state);
}

trait MulHOperation {
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4];
}
struct MulHOp;
struct MulHSuOp;
struct MulHUOp;
impl MulHOperation for MulHOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1) as i64;
        let rs2 = i32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
impl MulHOperation for MulHSuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1) as i64;
        let rs2 = u32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
impl MulHOperation for MulHUOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1) as i64;
        let rs2 = u32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
