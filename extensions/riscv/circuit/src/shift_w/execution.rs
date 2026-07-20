use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::ShiftWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{ShiftWLogicalExecutor, ShiftWRightArithmeticExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftWPreCompute {
    a: u8,
    b: u8,
    c: u8,
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
    ) -> Result<ShiftWOpcode, StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftWOpcode::from_usize(opcode.local_opcode_idx(self.offset()));
        if (shift_opcode == ShiftWOpcode::SRAW) != self.is_right_arithmetic() {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if inst.d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_REGISTER_AS
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftWPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(shift_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $shift_opcode:ident) => {
        match $shift_opcode {
            ShiftWOpcode::SLLW => Ok($execute_impl::<_, SllwOp>),
            ShiftWOpcode::SRLW => Ok($execute_impl::<_, SrlwOp>),
            ShiftWOpcode::SRAW => Ok($execute_impl::<_, SrawOp>),
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode)
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
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode)
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode)
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
        let data: &mut ShiftWPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode)
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode)
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: ShiftWOp>(
    pre_compute: &ShiftWPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2 =
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32);
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
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: ShiftWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &ShiftWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftWPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: ShiftWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
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
