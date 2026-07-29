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
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    core::{is_imm_opcode, is_reg_opcode, run_bitmanip_imm, run_bitmanip_reg, BITMANIP_OFFSET},
    BitManipImmExecutor, BitManipRegExecutor,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BitManipRegPreCompute {
    rs2_ptr: u8,
    rd_ptr: u8,
    rs1_ptr: u8,
    local_opcode: u8,
}

impl<A> BitManipRegExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BitManipRegPreCompute,
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
        if d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = opcode.local_opcode_idx(BITMANIP_OFFSET);
        if !is_reg_opcode(local_opcode) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BitManipRegPreCompute {
            rs2_ptr: c.as_canonical_u32() as u8,
            rd_ptr: a.as_canonical_u32() as u8,
            rs1_ptr: b.as_canonical_u32() as u8,
            local_opcode: local_opcode as u8,
        };
        Ok(())
    }
}

impl<F, A> InterpreterExecutor<F> for BitManipRegExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BitManipRegPreCompute>()
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
        let data: &mut BitManipRegPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_reg_e1_handler)
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
        let data: &mut BitManipRegPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_reg_e1_handler)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for BitManipRegExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BitManipRegPreCompute>>()
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
        let data: &mut E2PreCompute<BitManipRegPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_reg_e2_handler)
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
        let data: &mut E2PreCompute<BitManipRegPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_reg_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_reg_e12_impl<CTX: ExecutionCtxTrait>(
    pre_compute: &BitManipRegPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs1_ptr as u32);
    let rs2 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs2_ptr as u32);
    let rd = run_bitmanip_reg(
        pre_compute.local_opcode as usize,
        u64::from_le_bytes(rs1),
        u64::from_le_bytes(rs2),
    )
    .to_le_bytes();
    exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.rd_ptr as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_reg_e1_impl<CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BitManipRegPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BitManipRegPreCompute>()).borrow();
    execute_reg_e12_impl::<CTX>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_reg_e2_impl<CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BitManipRegPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BitManipRegPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_reg_e12_impl::<CTX>(&pre_compute.data, exec_state);
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BitManipImmPreCompute {
    imm: u8,
    rd_ptr: u8,
    rs1_ptr: u8,
    local_opcode: u8,
}

impl<A> BitManipImmExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BitManipImmPreCompute,
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
        if d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = opcode.local_opcode_idx(BITMANIP_OFFSET);
        if !is_imm_opcode(local_opcode) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BitManipImmPreCompute {
            imm: c.as_canonical_u32() as u8,
            rd_ptr: a.as_canonical_u32() as u8,
            rs1_ptr: b.as_canonical_u32() as u8,
            local_opcode: local_opcode as u8,
        };
        Ok(())
    }
}

impl<F, A> InterpreterExecutor<F> for BitManipImmExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BitManipImmPreCompute>()
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
        let data: &mut BitManipImmPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_imm_e1_handler)
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
        let data: &mut BitManipImmPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_imm_e1_handler)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for BitManipImmExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BitManipImmPreCompute>>()
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
        let data: &mut E2PreCompute<BitManipImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_imm_e2_handler)
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
        let data: &mut E2PreCompute<BitManipImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_imm_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_imm_e12_impl<CTX: ExecutionCtxTrait>(
    pre_compute: &BitManipImmPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs1_ptr as u32);
    let rd = run_bitmanip_imm(
        pre_compute.local_opcode as usize,
        u64::from_le_bytes(rs1),
        pre_compute.imm as u32,
    )
    .to_le_bytes();
    exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.rd_ptr as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_imm_e1_impl<CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BitManipImmPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BitManipImmPreCompute>()).borrow();
    execute_imm_e12_impl::<CTX>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_imm_e2_impl<CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BitManipImmPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BitManipImmPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_imm_e12_impl::<CTX>(&pre_compute.data, exec_state);
}
