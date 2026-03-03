use core::panic;

use openvm_instructions::{riscv::RV32_MEMORY_AS, LocalOpcode};
use openvm_instructions_derive::LocalOpcode;
use openvm_keccak256_guest::{KECCAKF_FUNCT7, OPCODE, XORIN_FUNCT3, XORIN_FUNCT7};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use openvm_decoder::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
pub enum KeccakfOpcode {
    KECCAKF,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x311]
#[repr(usize)]
pub enum XorinOpcode {
    XORIN,
}

#[derive(Default)]
pub struct Keccak256TranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Keccak256TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        // Safety note: KECCAKF_FUNCT3 == XORIN_FUNCT3 so it suffices to check once
        if (opcode, funct3) != (OPCODE, XORIN_FUNCT3) {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);

        if dec_insn.funct7 != XORIN_FUNCT7 as u32 && dec_insn.funct7 != KECCAKF_FUNCT7 as u32 {
            return None;
        }

        let instruction = if dec_insn.funct7 == XORIN_FUNCT7 as u32 {
            from_r_type(
                XorinOpcode::XORIN.global_opcode().as_usize(),
                RV32_MEMORY_AS as usize,
                &dec_insn,
                true,
            )
        } else {
            from_r_type(
                KeccakfOpcode::KECCAKF.global_opcode().as_usize(),
                RV32_MEMORY_AS as usize,
                &dec_insn,
                true,
            )
        };

        Some(TranspilerOutput::one_to_one(instruction))
    }
}
