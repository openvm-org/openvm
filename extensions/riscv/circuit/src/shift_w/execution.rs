use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::ShiftWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{ShiftWRightArithmeticExecutor, ShiftWLogicalExecutor};
#[allow(unused_imports)]
use crate::{adapters::imm_to_rv64_u64, common::*};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftWPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

trait ShiftWExecutorKind {
    fn offset(&self) -> usize;
    fn is_right_arithmetic(&self) -> bool;
}

impl<A> ShiftWExecutorKind for ShiftWLogicalExecutor<A> {
    fn offset(&self) -> usize {
        self.offset
    }

    fn is_right_arithmetic(&self) -> bool {
        false
    }
}

impl<A> ShiftWExecutorKind for ShiftWRightArithmeticExecutor<A> {
    fn offset(&self) -> usize {
        self.offset
    }

    fn is_right_arithmetic(&self) -> bool {
        true
    }
}

impl<T> ShiftWPreComputeExt for T where T: ShiftWExecutorKind {}

trait ShiftWPreComputeExt: ShiftWExecutorKind {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftWPreCompute,
    ) -> Result<(bool, ShiftWOpcode), StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftWOpcode::from_usize(opcode.local_opcode_idx(self.offset()));
        if (shift_opcode == ShiftWOpcode::SRAW) != self.is_right_arithmetic() {
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
        *data = ShiftWPreCompute {
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
    ($execute_impl:ident, $is_imm:ident, $shift_opcode:ident) => {
        match ($is_imm, $shift_opcode) {
            (true, ShiftWOpcode::SLLW) => Ok($execute_impl::<_, _, true, SllwOp>),
            (false, ShiftWOpcode::SLLW) => Ok($execute_impl::<_, _, false, SllwOp>),
            (true, ShiftWOpcode::SRLW) => Ok($execute_impl::<_, _, true, SrlwOp>),
            (false, ShiftWOpcode::SRLW) => Ok($execute_impl::<_, _, false, SrlwOp>),
            (true, ShiftWOpcode::SRAW) => Ok($execute_impl::<_, _, true, SrawOp>),
            (false, ShiftWOpcode::SRAW) => Ok($execute_impl::<_, _, false, SrawOp>),
        }
    };
}

impl<F, A> InterpreterExecutor<F> for ShiftWLogicalExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftWPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
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
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
    }
}

impl<F, A> InterpreterExecutor<F> for ShiftWRightArithmeticExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftWPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
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
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for ShiftWLogicalExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftWPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for ShiftWRightArithmeticExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftWPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV64_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftWOp,
>(
    pre_compute: &ShiftWPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV64_WORD_NUM_LIMBS] = if IS_IMM {
        pre_compute.c.to_le_bytes()[..RV64_WORD_NUM_LIMBS]
            .try_into()
            .unwrap()
    } else {
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32)
    };
    let rs2 = u32::from_le_bytes(rs2);

    let rd_word = u32::from_le_bytes(<OP as ShiftWOp>::compute(rs1, rs2));
    let rd = (rd_word as i32 as i64 as u64).to_le_bytes();
    exec_state.vm_write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        RV64_REGISTER_AS,
        pre_compute.a as u32,
        &rd,
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
    OP: ShiftWOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftWPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftWOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
}

trait ShiftWOp {
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: u32) -> [u8; RV64_WORD_NUM_LIMBS];
}
struct SllwOp;
struct SrlwOp;
struct SrawOp;
impl ShiftWOp for SllwOp {
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: u32) -> [u8; RV64_WORD_NUM_LIMBS] {
        let rs1 = u32::from_le_bytes(rs1);
        (rs1 << (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftWOp for SrlwOp {
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: u32) -> [u8; RV64_WORD_NUM_LIMBS] {
        let rs1 = u32::from_le_bytes(rs1);
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftWOp for SrawOp {
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: u32) -> [u8; RV64_WORD_NUM_LIMBS] {
        let rs1 = i32::from_le_bytes(rs1);
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}
