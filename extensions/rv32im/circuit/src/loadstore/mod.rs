mod core;

pub use core::*;

use openvm_circuit::{
    arch::{ExecutionError, InsExecutorE1, VmChipWrapper, VmExecutionState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::adapters::{Rv32LoadStoreAdapterChip, RV32_REGISTER_NUM_LIMBS};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> =
    VmChipWrapper<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<RV32_REGISTER_NUM_LIMBS>>;

impl<F> InsExecutorE1<F> for Rv32LoadStoreChip<F>
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
            c,
            g,
            f: enabled,
            ..
        } = instruction;

        // Get the local opcode for this instruction
        let local_opcode =
            Rv32LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let rs1_addr = b.as_canonical_u32();
        let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { state.memory.read(RV32_REGISTER_AS, rs1_addr) };
        let rs1_val = u32::from_le_bytes(rs1_bytes);

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        let shift_amount = ptr_val % 4;
        let ptr_val = ptr_val - shift_amount; // aligned ptr

        let read_bytes: [u8; RV32_REGISTER_NUM_LIMBS] = match local_opcode {
            Rv32LoadStoreOpcode::LOADW
            | Rv32LoadStoreOpcode::LOADB
            | Rv32LoadStoreOpcode::LOADH
            | Rv32LoadStoreOpcode::LOADBU
            | Rv32LoadStoreOpcode::LOADHU => {
                // For loads, read from memory
                unsafe { state.memory.read(RV32_MEMORY_AS, ptr_val) }
            }
            Rv32LoadStoreOpcode::STOREW
            | Rv32LoadStoreOpcode::STOREH
            | Rv32LoadStoreOpcode::STOREB => {
                // For stores, read the register value to be stored
                let rs2_addr = a.as_canonical_u32();
                unsafe { state.memory.read(RV32_REGISTER_AS, rs2_addr) }
            }
        };

        // For stores, we need the previous memory content to preserve unchanged bytes
        let prev_bytes: [u8; RV32_REGISTER_NUM_LIMBS] = match local_opcode {
            Rv32LoadStoreOpcode::STOREW
            | Rv32LoadStoreOpcode::STOREH
            | Rv32LoadStoreOpcode::STOREB => {
                // For stores, read current memory content
                unsafe { state.memory.read(RV32_MEMORY_AS, ptr_val) }
            }
            Rv32LoadStoreOpcode::LOADW
            | Rv32LoadStoreOpcode::LOADB
            | Rv32LoadStoreOpcode::LOADH
            | Rv32LoadStoreOpcode::LOADBU
            | Rv32LoadStoreOpcode::LOADHU => {
                // For loads, read current register content
                let rd_addr = a.as_canonical_u32();
                unsafe { state.memory.read(RV32_REGISTER_AS, rd_addr) }
            }
        };

        let read_data = read_bytes.map(F::from_canonical_u8);
        let prev_data = prev_bytes.map(F::from_canonical_u8);

        // Process the data according to the load/store type and alignment
        let write_data = run_write_data(local_opcode, read_data, prev_data, shift_amount);
        let write_bytes = write_data.map(|x| x.as_canonical_u32() as u8);

        if *enabled != F::ZERO {
            match local_opcode {
                Rv32LoadStoreOpcode::STOREW
                | Rv32LoadStoreOpcode::STOREH
                | Rv32LoadStoreOpcode::STOREB => {
                    // For stores, write to memory
                    unsafe { state.memory.write(RV32_MEMORY_AS, ptr_val, &write_data) };
                }
                Rv32LoadStoreOpcode::LOADW
                | Rv32LoadStoreOpcode::LOADB
                | Rv32LoadStoreOpcode::LOADH
                | Rv32LoadStoreOpcode::LOADBU
                | Rv32LoadStoreOpcode::LOADHU => {
                    let rd_addr = a.as_canonical_u32();
                    unsafe {
                        state.memory.write(RV32_REGISTER_AS, rd_addr, &write_bytes);
                    }
                }
            }
        }

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
