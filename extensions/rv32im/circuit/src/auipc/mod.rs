use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, NewVmChipWrapper, VmAirWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32AuipcOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::Rv32RdWriteAdapterAir;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcAir = VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>;
pub type Rv32AuipcChip<F> = NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcCoreChip>;

impl<F> InsExecutorE1<F>
    for NewVmChipWrapper<
        F,
        VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>,
        Rv32AuipcCoreChip,
    >
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
            Rv32AuipcOpcode::from_usize(opcode.local_opcode_idx(Rv32AuipcOpcode::CLASS_OFFSET));

        let imm = imm.as_canonical_u32();
        let rd_bytes = run_auipc(local_opcode, state.pc, imm);

        let rd_addr = a.as_canonical_u32();
        unsafe {
            state.memory.write(RV32_REGISTER_AS, rd_addr, &rd_bytes);
        }

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
