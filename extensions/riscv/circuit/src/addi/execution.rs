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
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};

use super::core::AddICoreExecutor;
use crate::adapters::{imm_to_u64, is_canonical_i12, U16_BITS};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct AddIPreCompute {
    imm: u64,
    rd_ptr: u8,
    rs1_ptr: u8,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> AddICoreExecutor<NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    pub(super) fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut AddIPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        let c = c.as_u32();
        if d.as_u32() != REGISTER_AS || e.as_u32() != IMM_AS || !is_canonical_i12(c) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = AddIPreCompute {
            imm: imm_to_u64(c),
            rd_ptr: a.as_u32() as u8,
            rs1_ptr: b.as_u32() as u8,
        };
        Ok(())
    }
}

impl<const NUM_LIMBS: usize> InterpreterExecutor for AddICoreExecutor<NUM_LIMBS, U16_BITS> {
    fn get_opcode_name(&self, opcode: usize) -> String {
        if NUM_LIMBS * U16_BITS == 32 {
            format!("{:?}", BaseAluWImmOpcode::from_usize(opcode - self.offset))
        } else {
            format!("{:?}", BaseAluImmOpcode::from_usize(opcode - self.offset))
        }
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AddIPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut AddIPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<Ctx, NUM_LIMBS>)
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
        let data: &mut AddIPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<Ctx, NUM_LIMBS>)
    }
}

impl<const NUM_LIMBS: usize> InterpreterMeteredExecutor for AddICoreExecutor<NUM_LIMBS, U16_BITS> {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AddIPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<AddIPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<Ctx, NUM_LIMBS>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<AddIPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<Ctx, NUM_LIMBS>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const NUM_LIMBS: usize>(
    pre_compute: &AddIPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = u64::from_le_bytes(
        exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.rs1_ptr as u32),
    );
    let rd = if NUM_LIMBS * U16_BITS == 32 {
        (rs1 as u32).wrapping_add(pre_compute.imm as u32) as i32 as i64 as u64
    } else {
        rs1.wrapping_add(pre_compute.imm)
    };
    exec_state.vm_write_bytes::<REGISTER_NUM_LIMBS>(
        REGISTER_AS,
        pre_compute.rd_ptr as u32,
        &rd.to_le_bytes(),
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const NUM_LIMBS: usize>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &AddIPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<AddIPreCompute>()).borrow();
    execute_e12_impl::<CTX, NUM_LIMBS>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const NUM_LIMBS: usize>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<AddIPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<AddIPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, NUM_LIMBS>(&pre_compute.data, exec_state);
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
    use openvm_riscv_transpiler::BaseAluImmOpcode;

    use super::*;
    use crate::AddIExecutor;

    fn addi_instruction(c: usize) -> Instruction {
        Instruction::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                REGISTER_NUM_LIMBS,
                2 * REGISTER_NUM_LIMBS,
                c,
                REGISTER_AS as usize,
                IMM_AS as usize,
            ],
        )
    }

    #[test]
    fn validates_canonical_i12_encoding() {
        let executor = AddIExecutor::new(BaseAluImmOpcode::CLASS_OFFSET);
        let mut data = AddIPreCompute {
            imm: 0,
            rd_ptr: 0,
            rs1_ptr: 0,
        };

        for c in [0, 0x7ff, 0xff_f800, 0xff_ffff] {
            assert!(executor
                .pre_compute_impl(0, &addi_instruction(c), &mut data)
                .is_ok());
        }

        for c in [0x800, 0xffff, 0x80_0000, 0x100_0000] {
            assert!(matches!(
                executor.pre_compute_impl(0, &addi_instruction(c), &mut data),
                Err(StaticProgramError::InvalidInstruction(0))
            ));
        }
    }
}
