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
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::ShiftRightArithmeticExecutor;
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftRightArithmeticPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> ShiftRightArithmeticExecutor<A, { BLOCK_FE_WIDTH }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftRightArithmeticPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if shift_opcode != ShiftOpcode::SRA {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if inst.d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_REGISTER_AS
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftRightArithmeticPreCompute {
            c: c.as_canonical_u32() as u64,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

macro_rules! dispatch {
    ($execute_impl:ident) => {
        Ok($execute_impl::<_, false>)
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftRightArithmeticExecutor<A, { BLOCK_FE_WIDTH }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftRightArithmeticPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftRightArithmeticPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler)
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
        let data: &mut ShiftRightArithmeticPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftRightArithmeticExecutor<A, { BLOCK_FE_WIDTH }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftRightArithmeticPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftRightArithmeticPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftRightArithmeticPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_IMM: bool>(
    pre_compute: &ShiftRightArithmeticPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32)
    };
    let rs1 = i64::from_le_bytes(rs1);
    let rs2 = u64::from_le_bytes(rs2);

    // Execute the arithmetic shift-right operation.
    // RV64: only the low 6 bits of rs2 are used for the shift amount.
    let rd = (rs1 >> (rs2 & 0x3F)).to_le_bytes();
    // Write the result back to memory
    exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_IMM: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &ShiftRightArithmeticPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftRightArithmeticPreCompute>())
            .borrow();
    execute_e12_impl::<CTX, IS_IMM>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const IS_IMM: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftRightArithmeticPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<ShiftRightArithmeticPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, IS_IMM>(&pre_compute.data, exec_state);
}
