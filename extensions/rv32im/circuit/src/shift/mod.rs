use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, NewVmChipWrapper, VmAirWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::adapters::{
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32ShiftAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32ShiftStep =
    ShiftStep<Rv32BaseAluAdapterStep<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32ShiftChip<F> = NewVmChipWrapper<F, Rv32ShiftAir, Rv32ShiftStep>;

impl<F> InsExecutorE1<F> for Rv32ShiftChip<F>
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
            opcode, a, b, c, e, ..
        } = instruction;

        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.air.core.offset));

        let rs1_addr = b.as_canonical_u32();
        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };

        let rs2_bytes = if e.as_canonical_u32() == RV32_IMM_AS {
            // Use immediate value
            let imm = c.as_canonical_u32();
            imm.to_le_bytes()
        } else {
            // Read from register
            let rs2_addr = c.as_canonical_u32();
            let rs2_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
                unsafe { state.memory.read(RV32_REGISTER_AS, rs2_addr) };
            rs2_bytes
        };

        // Execute the shift operation
        let (rd_bytes, _, _) = run_shift::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            shift_opcode,
            &rs1_bytes,
            &rs2_bytes,
        );

        let rd_addr = a.as_canonical_u32();
        unsafe {
            state.memory.write(RV32_REGISTER_AS, rd_addr, &rd_bytes);
        }

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
