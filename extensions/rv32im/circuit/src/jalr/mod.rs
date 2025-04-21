use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, VmChipWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalrOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::Rv32JalrAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrAdapterChip<F>, Rv32JalrCoreChip>;

impl<F> InsExecutorE1<F> for Rv32JalrChip<F>
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
            b,
            a,
            c,
            f: enabled,
            g,
            ..
        } = instruction;

        let local_opcode =
            Rv32JalrOpcode::from_usize(opcode.local_opcode_idx(Rv32JalrOpcode::CLASS_OFFSET));

        let rs1_addr = b.as_canonical_u32();
        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };

        // TODO(ayush): directly read as u32 from memory
        let rs1_bytes = u32::from_le_bytes(rs1_bytes);

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        // TODO(ayush): should this be [u8; 4]?
        let (to_pc, rd_bytes) = run_jalr(local_opcode, state.pc, imm_extended, rs1_bytes);
        let rd_bytes = rd_bytes.map(|x| x as u8);

        // TODO(ayush): do i need this enabled check?
        if *enabled != F::ZERO {
            let rd_addr = a.as_canonical_u32();
            unsafe {
                state.memory.write(RV32_REGISTER_AS, rd_addr, &rd_bytes);
            }
        }

        state.pc = to_pc;

        Ok(())
    }
}
