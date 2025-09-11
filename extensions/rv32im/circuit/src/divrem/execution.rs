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
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::DivRemExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DivRemPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DivRemPreCompute,
    ) -> Result<DivRemOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let pre_compute: &mut DivRemPreCompute = data.borrow_mut();
        *pre_compute = DivRemPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            DivRemOpcode::DIV => Ok($execute_impl::<_, _, DivOp>),
            DivRemOpcode::DIVU => Ok($execute_impl::<_, _, DivuOp>),
            DivRemOpcode::REM => Ok($execute_impl::<_, _, RemOp>),
            DivRemOpcode::REMU => Ok($execute_impl::<_, _, RemuOp>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> Executor<F>
    for DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<DivRemPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
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
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DivRemPreCompute>>()
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
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
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
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: &DivRemPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let result = <OP as DivRemOp>::compute(rs1, rs2);
    exec_state.vm_write::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32, &result);
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DivRemPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DivRemPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, instret, pc, exec_state);
}

trait DivRemOp {
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4];
}
struct DivOp;
struct DivuOp;
struct RemOp;
struct RemuOp;

impl DivRemOp for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => [u8::MAX; 4],
            (i32::MIN, -1) => rs1,
            _ => (rs1_i32 / rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemOp for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        if rs2 == [0; 4] {
            [u8::MAX; 4]
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 / rs2).to_le_bytes()
        }
    }
}

impl DivRemOp for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => rs1,
            (i32::MIN, -1) => [0; 4],
            _ => (rs1_i32 % rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemOp for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        if rs2 == [0; 4] {
            rs1
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 % rs2).to_le_bytes()
        }
    }
}
