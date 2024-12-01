use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, utils::isize_to_field, BaseAluOpcode,
    BranchEqualOpcode, LessThanOpcode, MulOpcode, Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode,
    Rv32LessThan256Opcode, Rv32Mul256Opcode, Rv32Shift256Opcode, ShiftOpcode, UsizeOpcode,
};
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::{BType, RType};
use strum_macros::FromRepr;

#[derive(Default)]
pub struct Int256TranspilerExtension;

// TODO: the opcode and func3 will be imported from `guest` crate
pub(crate) const OPCODE: u8 = 0x0b;
pub(crate) const INT256_FUNCT3: u8 = 0b101;
pub(crate) const BEQ256_FUNCT3: u8 = 0b110;

// TODO: this should be moved to `guest` crate
/// funct7 options for 256-bit integer instructions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum Int256Funct7 {
    Add = 0,
    Sub,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
    Mul,
}

impl<F: PrimeField32> TranspilerExtension<F> for Int256TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != OPCODE {
            return None;
        }
        if funct3 != INT256_FUNCT3 && funct3 != BEQ256_FUNCT3 {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let instruction = match funct3 {
            INT256_FUNCT3 => {
                let global_opcode = match Int256Funct7::from_repr(dec_insn.funct7 as u8) {
                    Some(Int256Funct7::Add) => {
                        BaseAluOpcode::ADD as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sub) => {
                        BaseAluOpcode::SUB as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Xor) => {
                        BaseAluOpcode::XOR as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Or) => {
                        BaseAluOpcode::OR as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::And) => {
                        BaseAluOpcode::AND as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sll) => {
                        ShiftOpcode::SLL as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Srl) => {
                        ShiftOpcode::SRL as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sra) => {
                        ShiftOpcode::SRA as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Slt) => {
                        LessThanOpcode::SLT as usize + Rv32LessThan256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sltu) => {
                        LessThanOpcode::SLTU as usize + Rv32LessThan256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Mul) => {
                        MulOpcode::MUL as usize + Rv32Mul256Opcode::default_offset()
                    }
                    _ => unimplemented!(),
                };
                Some(from_r_type(global_opcode, 2, &dec_insn))
            }
            BEQ256_FUNCT3 => {
                let dec_insn = BType::new(instruction_u32);
                Some(Instruction::new(
                    BranchEqualOpcode::BEQ as usize + Rv32BranchEqual256Opcode::default_offset(),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    isize_to_field(dec_insn.imm as isize),
                    F::ONE,
                    F::TWO,
                    F::ZERO,
                    F::ZERO,
                ))
            }
            _ => None,
        };
        instruction.map(|instruction| (instruction, 1))
    }
}
