use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_u_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::UType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x330]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32MemcpyOpcode {
    MEMCPY_LOOP,
}

// pub const OPCODE: u8 = 0x0b;
// pub const KECCAK256_FUNCT3: u8 = 0b100;
// pub const KECCAK256_FUNCT7: u8 = 0;
// Custom opcode for memcpy_loop instruction
pub const MEMCPY_LOOP_OPCODE: u8 = 0x73; // Custom opcode
pub const MEMCPY_LOOP_FUNCT3: u8 = 0x0; // Custom funct3

#[derive(Default)]
pub struct MemcpyTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for MemcpyTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }

        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        // Check if this is our custom memcpy_loop instruction
        if (opcode, funct3) != (MEMCPY_LOOP_OPCODE, MEMCPY_LOOP_FUNCT3) {
            return None;
        }

        // Parse I-type instruction format
        let dec_insn = UType::new(instruction_u32);
        let shift = dec_insn.imm as u8;

        // Validate shift value (0, 1, 2, or 3)
        if shift > 3u8 {
            return None;
        }

        // Convert to OpenVM instruction format
        let instruction = from_u_type(
            Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode().as_usize(),
            &dec_insn,
        );

        Some(TranspilerOutput::one_to_one(instruction))
    }
}
