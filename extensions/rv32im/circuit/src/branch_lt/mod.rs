use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, VmChipWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::Rv32BranchAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanChip<F> = VmChipWrapper<
    F,
    Rv32BranchAdapterChip<F>,
    BranchLessThanCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

impl<F> InsExecutorE1<F> for Rv32BranchLessThanChip<F>
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
    {
        let Instruction {
            opcode,
            a,
            b,
            c: imm,
            ..
        } = instruction;

        let blt_opcode = BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let rs1_addr = a.as_canonical_u32();
        let rs2_addr = b.as_canonical_u32();

        // TODO(ayush): why even have RV32_REGISTER_NUM_LIMBS when it is equal to RV32_REGISTER_NUM_LIMBS?
        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };
        let rs2_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs2_addr) };

        // TODO(ayush): why is this conversion necessary?
        let rs1_bytes: [u32; RV32_REGISTER_NUM_LIMBS] = rs1_bytes.map(|x| x as u32);
        let rs2_bytes: [u32; RV32_REGISTER_NUM_LIMBS] = rs2_bytes.map(|y| y as u32);

        let (cmp_result, _, _, _) =
            run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(blt_opcode, &rs1_bytes, &rs2_bytes);

        if cmp_result {
            let imm = imm.as_canonical_u32();
            state.pc = state.pc.wrapping_add(imm);
        } else {
            state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }
}
