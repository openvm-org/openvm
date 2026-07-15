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
use openvm_riscv_transpiler::BaseAluWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::AddSubWExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct AddSubWPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A> AddSubWExecutor<A> {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AddSubWPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        if d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = AddSubWPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match BaseAluWOpcode::from_usize($opcode.local_opcode_idx($offset)) {
                BaseAluWOpcode::ADDW => $execute_impl::<_, AddwOp>,
                BaseAluWOpcode::SUBW => $execute_impl::<_, SubwOp>,
            },
        )
    };
}

impl<F, A> InterpreterExecutor<F> for AddSubWExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AddSubWPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut AddSubWPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
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
        let data: &mut AddSubWPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for AddSubWExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AddSubWPreCompute>>()
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
        let data: &mut E2PreCompute<AddSubWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<AddSubWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: AluWOp>(
    pre_compute: &AddSubWPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2 =
        exec_state.vm_read_bytes::<RV64_WORD_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32);

    let rs1_low = u32::from_le_bytes(rs1);
    let rs2_low = u32::from_le_bytes(rs2);
    let rd_word = <OP as AluWOp>::compute(rs1_low, rs2_low);
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
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: AluWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &AddSubWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<AddSubWPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: AluWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<AddSubWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<AddSubWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait AluWOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct AddwOp;
struct SubwOp;
impl AluWOp for AddwOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_add(rs2)
    }
}
impl AluWOp for SubwOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_sub(rs2)
    }
}
