use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::DivRemExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DivRemPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> DivRemExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS> {
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
        if d.as_canonical_u32() != RV64_REGISTER_AS {
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
            DivRemOpcode::DIV => Ok($execute_impl::<_, DivOp>),
            DivRemOpcode::DIVU => Ok($execute_impl::<_, DivuOp>),
            DivRemOpcode::REM => Ok($execute_impl::<_, RemOp>),
            DivRemOpcode::REMU => Ok($execute_impl::<_, RemuOp>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for DivRemExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
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
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for DivRemExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
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
    ) -> Result<Handler<Ctx>, StaticProgramError>
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
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: &DivRemPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.c as u32);
    let result = <OP as DivRemOp>::compute(rs1, rs2);
    exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.a as u32, &result);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &DivRemPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<DivRemPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DivRemPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<DivRemPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait DivRemOp {
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8];
}
struct DivOp;
struct DivuOp;
struct RemOp;
struct RemuOp;

impl DivRemOp for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1_i64 = i64::from_le_bytes(rs1);
        let rs2_i64 = i64::from_le_bytes(rs2);
        match (rs1_i64, rs2_i64) {
            (_, 0) => [u8::MAX; 8],
            (i64::MIN, -1) => rs1,
            _ => (rs1_i64 / rs2_i64).to_le_bytes(),
        }
    }
}

impl DivRemOp for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        if rs2 == [0; 8] {
            [u8::MAX; 8]
        } else {
            let rs1 = u64::from_le_bytes(rs1);
            let rs2 = u64::from_le_bytes(rs2);
            (rs1 / rs2).to_le_bytes()
        }
    }
}

impl DivRemOp for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1_i64 = i64::from_le_bytes(rs1);
        let rs2_i64 = i64::from_le_bytes(rs2);
        match (rs1_i64, rs2_i64) {
            (_, 0) => rs1,
            (i64::MIN, -1) => [0; 8],
            _ => (rs1_i64 % rs2_i64).to_le_bytes(),
        }
    }
}

impl DivRemOp for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        if rs2 == [0; 8] {
            rs1
        } else {
            let rs1 = u64::from_le_bytes(rs1);
            let rs2 = u64::from_le_bytes(rs2);
            (rs1 % rs2).to_le_bytes()
        }
    }
}
