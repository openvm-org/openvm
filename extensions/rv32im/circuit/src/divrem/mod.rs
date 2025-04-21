use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, VmChipWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::adapters::{Rv32MultAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32DivRemChip<F> = VmChipWrapper<
    F,
    Rv32MultAdapterChip<F>,
    DivRemCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

impl<F> InsExecutorE1<F> for Rv32DivRemChip<F>
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
            opcode, a, b, c, ..
        } = *instruction;

        // Determine opcode and operation type
        let divrem_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        // Read input registers
        let rs1_addr = b.as_canonical_u32();
        let rs2_addr = c.as_canonical_u32();

        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };
        let rs2_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs2_addr) };

        // TODO(ayush): remove this conversion
        let rs1_bytes = rs1_bytes.map(|x| x as u32);
        let rs2_bytes = rs2_bytes.map(|y| y as u32);

        let is_div = divrem_opcode == DivRemOpcode::DIV || divrem_opcode == DivRemOpcode::DIVU;
        let is_signed = divrem_opcode == DivRemOpcode::DIV || divrem_opcode == DivRemOpcode::REM;

        // Perform division/remainder computation
        let (q, r, _, _, _, _) = run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            is_signed, &rs1_bytes, &rs2_bytes,
        );

        // Determine result based on operation type (DIV or REM)
        let rd_bytes = if is_div {
            q.map(|x| x as u8)
        } else {
            r.map(|x| x as u8)
        };

        // Write result to destination register
        let rd_addr = a.as_canonical_u32();
        unsafe {
            state.memory.write(RV32_REGISTER_AS, rd_addr, &rd_bytes);
        }

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
