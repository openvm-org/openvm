use openvm_instructions::{instruction::Instruction, UsizeOpcode};
use openvm_instructions_derive::UsizeOpcode;
use openvm_keccak256_guest::{FUNCT3, OPCODE};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
pub enum Rv32KeccakOpcode {
    KECCAK256,
}

#[derive(Default)]
pub struct Keccak256TranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Keccak256TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if (opcode, funct3) != (OPCODE, FUNCT3) {
            return None;
        }
        let dec_insn = RType::new(instruction_u32);
        let instruction = from_r_type(
            Rv32KeccakOpcode::KECCAK256.with_default_offset(),
            2,
            &dec_insn,
        );
        Some((instruction, 1))
    }
}
