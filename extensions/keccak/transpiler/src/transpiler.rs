use axvm_instructions::{instruction::Instruction, Rv32KeccakOpcode, UsizeOpcode};
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::RType;

#[derive(Default)]
pub struct KeccakTranspilerExtension;

// TODO: the opcode and func3 will be imported from `guest` crate
pub(crate) const OPCODE: u8 = 0x0b;
pub(crate) const FUNCT3: u8 = 0b100;

impl<F: PrimeField32> TranspilerExtension<F> for KeccakTranspilerExtension {
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
