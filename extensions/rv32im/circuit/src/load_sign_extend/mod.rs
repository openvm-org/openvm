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

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::Rv32LoadStoreAdapterChip;

mod core;
pub use core::*;
use std::array;

#[cfg(test)]
mod tests;

pub type Rv32LoadSignExtendChip<F> = VmChipWrapper<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadSignExtendCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

impl<F> InsExecutorE1<F> for Rv32LoadSignExtendChip<F>
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
            f: enabled,
            g,
            ..
        } = instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

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
            Rv32LoadStoreOpcode::LOADB | Rv32LoadStoreOpcode::LOADH => unsafe {
                state.memory.read(RV32_MEMORY_AS, ptr_val)
            },
            _ => unreachable!("Only LOADB and LOADH are supported by LoadSignExtendCoreChip chip"),
        };
        let read_data: [F; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|i| F::from_canonical_u8(read_bytes[i]));

        // TODO(ayush): clean this up for e1
        let write_data = run_write_data_sign_extend::<_, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            local_opcode,
            read_data,
            [F::ZERO; RV32_REGISTER_NUM_LIMBS],
            shift_amount,
        );
        let write_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|i| write_data[i].as_canonical_u32() as u8);

        // Only proceed if instruction is enabled
        if *enabled != F::ZERO {
            // Write result to destination register
            let rd_addr = a.as_canonical_u32();
            unsafe {
                state.memory.write(RV32_REGISTER_AS, rd_addr, &write_bytes);
            }
        }

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
