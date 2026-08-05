use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{IMM_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{ShiftImmOpcode, ShiftWImmOpcode};

use super::ShiftRightArithmeticImmCoreExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftRightArithmeticImmPreCompute {
    shamt: u8,
    rd_ptr: u8,
    rs1_ptr: u8,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftRightArithmeticImmCoreExecutor<NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut ShiftRightArithmeticImmPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let shamt = c.as_u32();
        if opcode.local_opcode_idx(self.offset) != ShiftImmOpcode::SRAI as usize
            || d.as_u32() != REGISTER_AS
            || e.as_u32() != IMM_AS
            || shamt >= (NUM_LIMBS * LIMB_BITS) as u32
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftRightArithmeticImmPreCompute {
            shamt: shamt as u8,
            rd_ptr: a.as_u32() as u8,
            rs1_ptr: b.as_u32() as u8,
        };
        Ok(())
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor
    for ShiftRightArithmeticImmCoreExecutor<NUM_LIMBS, LIMB_BITS>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if NUM_LIMBS * LIMB_BITS == 32 {
            format!("{:?}", ShiftWImmOpcode::from_usize(opcode - self.offset))
        } else {
            format!("{:?}", ShiftImmOpcode::from_usize(opcode - self.offset))
        }
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftRightArithmeticImmPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftRightArithmeticImmPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut ShiftRightArithmeticImmPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<Ctx, NUM_LIMBS, LIMB_BITS>)
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor
    for ShiftRightArithmeticImmCoreExecutor<NUM_LIMBS, LIMB_BITS>
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftRightArithmeticImmPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftRightArithmeticImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftRightArithmeticImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<Ctx, NUM_LIMBS, LIMB_BITS>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: &ShiftRightArithmeticImmPreCompute,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) {
    let rs1 =
        exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.rs1_ptr as u32);
    let rd = if NUM_LIMBS * LIMB_BITS == 32 {
        ((u32::from_le_bytes(rs1[..4].try_into().unwrap()) as i32) >> pre_compute.shamt) as i64
    } else {
        i64::from_le_bytes(rs1) >> pre_compute.shamt
    }
    .to_le_bytes();
    exec_state.vm_write_bytes(REGISTER_AS, pre_compute.rd_ptr as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) {
    let pre_compute: &ShiftRightArithmeticImmPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftRightArithmeticImmPreCompute>())
            .borrow();
    execute_e12_impl::<Ctx, NUM_LIMBS, LIMB_BITS>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    Ctx: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) {
    let pre_compute: &E2PreCompute<ShiftRightArithmeticImmPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<ShiftRightArithmeticImmPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<Ctx, NUM_LIMBS, LIMB_BITS>(&pre_compute.data, exec_state);
}
