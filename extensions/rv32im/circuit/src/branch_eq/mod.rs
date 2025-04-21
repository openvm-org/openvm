use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, VmChipWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::Rv32BranchAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> =
    VmChipWrapper<F, Rv32BranchAdapterChip<F>, BranchEqualCoreChip<RV32_REGISTER_NUM_LIMBS>>;

impl<F> InsExecutorE1<F> for Rv32BranchEqualChip<F>
where
    F: PrimeField32,
{
    fn execute_e1<Mem, Ctx>(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        Mem: GuestMemory,
        F: PrimeField32,
    {
        let Instruction {
            opcode,
            a,
            b,
            c: imm,
            ..
        } = instruction;

        let branch_eq_opcode =
            BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.core.air.offset));

        let rs1_addr = a.as_canonical_u32();
        let rs2_addr = b.as_canonical_u32();

        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };
        let rs2_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs2_addr) };

        // TODO(ayush): avoid this conversion
        let rs1_bytes: [u32; RV32_REGISTER_NUM_LIMBS] = rs1_bytes.map(|x| x as u32);
        let rs2_bytes: [u32; RV32_REGISTER_NUM_LIMBS] = rs2_bytes.map(|y| y as u32);

        // TODO(ayush): probably don't need the other values
        let (cmp_result, _, _) =
            run_eq::<F, RV32_REGISTER_NUM_LIMBS>(branch_eq_opcode, &rs1_bytes, &rs2_bytes);

        if cmp_result {
            let imm = imm.as_canonical_u32();
            state.pc = state.pc.wrapping_add(imm);
        } else {
            // TODO(ayush): why not DEFAULT_PC_STEP or some constant?
            state.pc = state.pc.wrapping_add(self.air.pc_step);
        }

        Ok(())
    }
}
