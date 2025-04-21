use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, NewVmChipWrapper, VmAirWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{Rv32CondRdWriteAdapterAir, RV_J_TYPE_IMM_BITS};

use crate::adapters::Rv32CondRdWriteAdapterAir;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalLuiAir = VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>;
pub type Rv32JalLuiChip<F> = NewVmChipWrapper<F, Rv32JalLuiAir, Rv32JalLuiCoreChip>;

impl<F> InsExecutorE1<F> for Rv32JalLuiChip<F>
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
            opcode, a, c: imm, ..
        } = instruction;

        let local_opcode =
            Rv32JalLuiOpcode::from_usize(opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET));

        let imm = imm.as_canonical_u32();
        let signed_imm = match local_opcode {
            JAL => (imm + (1 << (RV_J_TYPE_IMM_BITS - 1))) as i32 - (1 << (RV_J_TYPE_IMM_BITS - 1)),
            LUI => imm as i32,
        };

        let (to_pc, rd_bytes) = run_jal_lui(local_opcode, state.pc, signed_imm);

        let rd_addr = a.as_canonical_u32();
        unsafe {
            state.memory.write(RV32_REGISTER_AS, rd_addr, &rd_bytes);
        }

        state.pc = to_pc;

        Ok(())
    }
}
