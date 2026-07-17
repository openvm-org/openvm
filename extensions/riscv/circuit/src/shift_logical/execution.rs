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
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::ShiftLogicalExecutor;
use crate::adapters::imm_to_rv64_u64;
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftLogicalPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftLogicalExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftLogicalPreCompute,
    ) -> Result<(bool, ShiftOpcode), StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if shift_opcode == ShiftOpcode::SRA {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let e_u32 = e.as_canonical_u32();
        if inst.d.as_canonical_u32() != RV64_REGISTER_AS
            || !(e_u32 == RV64_IMM_AS || e_u32 == RV64_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV64_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = ShiftLogicalPreCompute {
            c: if is_imm {
                imm_to_rv64_u64(c_u32)
            } else {
                c_u32 as u64
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        // `d` is always expected to be RV64_REGISTER_AS.
        Ok((is_imm, shift_opcode))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $shift_opcode:ident, $pc:ident) => {
        match ($is_imm, $shift_opcode) {
            (true, ShiftOpcode::SLL) => Ok($execute_impl::<_, true, SllOp>),
            (false, ShiftOpcode::SLL) => Ok($execute_impl::<_, false, SllOp>),
            (true, ShiftOpcode::SRL) => Ok($execute_impl::<_, true, SrlOp>),
            (false, ShiftOpcode::SRL) => Ok($execute_impl::<_, false, SrlOp>),
            (_, ShiftOpcode::SRA) => Err(StaticProgramError::InvalidInstruction($pc)),
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftLogicalExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftLogicalPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftLogicalPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode, pc)
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
        let data: &mut ShiftLogicalPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode, pc)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftLogicalExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftLogicalPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftLogicalPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode, pc)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftLogicalPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode, pc)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_IMM: bool, OP: ShiftOp>(
    pre_compute: &ShiftLogicalPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32)
    };
    let rs2 = u64::from_le_bytes(rs2);

    // Execute the shift operation
    let rd = <OP as ShiftOp>::compute(rs1, rs2);
    // Write the result back to memory
    exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_IMM: bool, OP: ShiftOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &ShiftLogicalPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftLogicalPreCompute>()).borrow();
    execute_e12_impl::<CTX, IS_IMM, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const IS_IMM: bool, OP: ShiftOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftLogicalPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<ShiftLogicalPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
}

trait ShiftOp {
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS];
}
struct SllOp;
struct SrlOp;
impl ShiftOp for SllOp {
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        // RV64: only the low 6 bits of rs2 are used for the shift amount.
        (rs1 << (rs2 & 0x3F)).to_le_bytes()
    }
}
impl ShiftOp for SrlOp {
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        // RV64: only the low 6 bits of rs2 are used for the shift amount.
        (rs1 >> (rs2 & 0x3F)).to_le_bytes()
    }
}
