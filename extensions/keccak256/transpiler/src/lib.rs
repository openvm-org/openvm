use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use openvm_keccak256_guest::{KECCAK256_FUNCT3, KECCAK256_FUNCT7, OPCODE};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
pub enum Rv32KeccakOpcode {
    KECCAK256,
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

        if (opcode, funct3) != (OPCODE, KECCAK256_FUNCT3) {
            return None;
        }
        let dec_insn = RType::new(instruction_u32);
        if dec_insn.funct7 != KECCAK256_FUNCT7 as u32 {
            return None;
        }
        let instruction = from_r_type(
            Rv32KeccakOpcode::KECCAK256.global_opcode().as_usize(),
            2,
            &dec_insn,
            true,
        );
        Some(TranspilerOutput::one_to_one(instruction))
    }
}
