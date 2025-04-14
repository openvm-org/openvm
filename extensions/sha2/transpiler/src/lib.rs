use openvm_instructions::{riscv::RV32_MEMORY_AS, LocalOpcode};
use openvm_instructions_derive::LocalOpcode;
use openvm_sha2_guest::{Sha2BaseFunct7, OPCODE, SHA2_FUNCT3};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x320]
#[repr(usize)]
pub enum Rv32Sha2Opcode {
    SHA256,
    SHA512,
    SHA384,
}

#[derive(Default)]
pub struct Sha2TranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Sha2TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if (opcode, funct3) != (OPCODE, SHA2_FUNCT3) {
            return None;
        }
        let dec_insn = RType::new(instruction_u32);

        if dec_insn.funct7 == Sha2BaseFunct7::Sha256 as u32 {
            let instruction = from_r_type(
                Rv32Sha2Opcode::SHA256.global_opcode().as_usize(),
                RV32_MEMORY_AS as usize,
                &dec_insn,
                true,
            );
            Some(TranspilerOutput::one_to_one(instruction))
        } else if dec_insn.funct7 == Sha2BaseFunct7::Sha512 as u32 {
            let instruction = from_r_type(
                Rv32Sha2Opcode::SHA512.global_opcode().as_usize(),
                RV32_MEMORY_AS as usize,
                &dec_insn,
                true,
            );
            Some(TranspilerOutput::one_to_one(instruction))
        } else if dec_insn.funct7 == Sha2BaseFunct7::Sha384 as u32 {
            let instruction = from_r_type(
                Rv32Sha2Opcode::SHA384.global_opcode().as_usize(),
                RV32_MEMORY_AS as usize,
                &dec_insn,
                true,
            );
            Some(TranspilerOutput::one_to_one(instruction))
        } else {
            None
        }
    }
}
