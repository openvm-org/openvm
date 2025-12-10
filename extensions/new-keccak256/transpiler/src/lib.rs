use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use openvm_new_keccak256_guest::{
    KECCAKF_FUNCT3, KECCAKF_FUNCT7, OPCODE, XORIN_FUNCT3, XORIN_FUNCT7,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
pub enum Rv32NewKeccakOpcode {
    KECCAKF,
    XORIN,
}

#[derive(Default)]
pub struct NewKeccakTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for NewKeccakTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        // Safety note: KECCAKF_FUNCT3 == XORIN_FUNCT3 so it suffices to check once
        if (opcode, funct3) != (OPCODE, KECCAKF_FUNCT3) {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);

        if dec_insn.funct7 != KECCAKF_FUNCT7 as u32 && dec_insn.funct7 != XORIN_FUNCT7 as u32 {
            return None;
        }

        // TODO: what does e_as mean?
        let instruction = from_r_type(
            if dec_insn.funct7 == KECCAKF_FUNCT7 as u32 {
                Rv32NewKeccakOpcode::KECCAKF.global_opcode().as_usize()
            } else {
                Rv32NewKeccakOpcode::XORIN.global_opcode().as_usize()
            },
            2,
            &dec_insn,
            true,
        );

        Some(TranspilerOutput::one_to_one(instruction))
    }
}
