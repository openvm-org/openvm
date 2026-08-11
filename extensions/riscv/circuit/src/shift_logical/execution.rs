use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::ShiftLogicalCoreExecutor;
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftLogicalPreCompute {
    rs2_ptr: u8,
    a: u8,
    b: u8,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftLogicalCoreExecutor<NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut ShiftLogicalPreCompute,
    ) -> Result<ShiftOpcode, StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if shift_opcode == ShiftOpcode::SRA {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if inst.d.as_u32() != REGISTER_AS || e.as_u32() != REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftLogicalPreCompute {
            rs2_ptr: c.as_u32() as u8,
            a: a.as_u32() as u8,
            b: b.as_u32() as u8,
        };
        Ok(shift_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $shift_opcode:ident, $pc:ident) => {
        match $shift_opcode {
            ShiftOpcode::SLL => Ok($execute_impl::<_, SllOp>),
            ShiftOpcode::SRL => Ok($execute_impl::<_, SrlOp>),
            ShiftOpcode::SRA => Err(StaticProgramError::InvalidInstruction($pc)),
        }
    };
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftLogicalCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.offset))
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftLogicalPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut ShiftLogicalPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode, pc)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut ShiftLogicalPreCompute = data.borrow_mut();
        let shift_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, shift_opcode, pc)
    }
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftLogicalCoreExecutor<NUM_LIMBS, LIMB_BITS>
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
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftLogicalPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode, pc)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftLogicalPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let shift_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, shift_opcode, pc)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: &ShiftLogicalPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.b as u32);
    let rs2 =
        exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.rs2_ptr as u32);
    let rs2 = u64::from_le_bytes(rs2);

    // Execute the shift operation
    let rd = <OP as ShiftOp>::compute(rs1, rs2);
    // Write the result back to memory
    exec_state.vm_write_bytes(REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &ShiftLogicalPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftLogicalPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: ShiftOp>(
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
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait ShiftOp {
    fn compute(rs1: [u8; REGISTER_NUM_LIMBS], rs2: u64) -> [u8; REGISTER_NUM_LIMBS];
}
struct SllOp;
struct SrlOp;
impl ShiftOp for SllOp {
    fn compute(rs1: [u8; REGISTER_NUM_LIMBS], rs2: u64) -> [u8; REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        // RV64: only the low 6 bits of rs2 are used for the shift amount.
        (rs1 << (rs2 & 0x3F)).to_le_bytes()
    }
}
impl ShiftOp for SrlOp {
    fn compute(rs1: [u8; REGISTER_NUM_LIMBS], rs2: u64) -> [u8; REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        // RV64: only the low 6 bits of rs2 are used for the shift amount.
        (rs1 >> (rs2 & 0x3F)).to_le_bytes()
    }
}
