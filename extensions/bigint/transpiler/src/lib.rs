use openvm_bigint_guest::{Int256Funct7, BEQ256_FUNCT3, INT256_FUNCT3, OPCODE};
use openvm_decoder::instruction_formats::{BType, RType};
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    riscv::REGISTER_NUM_LIMBS,
    LocalOpcode, VmOpcode,
};
use openvm_instructions_derive::LocalOpcode;
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use strum::IntoEnumIterator;

// =================================================================================================
// Intrinsics: 256-bit Integers
// =================================================================================================

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x400]
pub struct BaseAlu256Opcode(pub BaseAluOpcode);

impl BaseAlu256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BaseAluOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x405]
pub struct Shift256Opcode(pub ShiftOpcode);

impl Shift256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        ShiftOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x408]
pub struct LessThan256Opcode(pub LessThanOpcode);

impl LessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        LessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x420]
pub struct BranchEqual256Opcode(pub BranchEqualOpcode);

impl BranchEqual256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x425]
pub struct BranchLessThan256Opcode(pub BranchLessThanOpcode);

impl BranchLessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchLessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x450]
pub struct Mul256Opcode(pub MulOpcode);

impl Mul256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        MulOpcode::iter().map(Self)
    }
}

#[derive(Default)]
pub struct Int256TranspilerExtension;

impl TranspilerExtension for Int256TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput> {
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
                        BaseAluOpcode::ADD as usize + BaseAlu256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Sub) => {
                        BaseAluOpcode::SUB as usize + BaseAlu256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Xor) => {
                        BaseAluOpcode::XOR as usize + BaseAlu256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Or) => {
                        BaseAluOpcode::OR as usize + BaseAlu256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::And) => {
                        BaseAluOpcode::AND as usize + BaseAlu256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Sll) => {
                        ShiftOpcode::SLL as usize + Shift256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Srl) => {
                        ShiftOpcode::SRL as usize + Shift256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Sra) => {
                        ShiftOpcode::SRA as usize + Shift256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Slt) => {
                        LessThanOpcode::SLT as usize + LessThan256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Sltu) => {
                        LessThanOpcode::SLTU as usize + LessThan256Opcode::CLASS_OFFSET
                    }
                    Some(Int256Funct7::Mul) => MulOpcode::MUL as usize + Mul256Opcode::CLASS_OFFSET,
                    _ => unimplemented!(),
                };
                Some(from_r_type(global_opcode, 2, &dec_insn, true))
            }
            BEQ256_FUNCT3 => {
                let dec_insn = BType::new(instruction_u32);
                Some(Instruction::new(
                    VmOpcode::from_usize(
                        BranchEqualOpcode::BEQ.local_usize() + BranchEqual256Opcode::CLASS_OFFSET,
                    ),
                    InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
                    InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
                    InstructionOperand::from_i32(dec_insn.imm),
                    InstructionOperand::ONE,
                    InstructionOperand::TWO,
                    InstructionOperand::ZERO,
                    InstructionOperand::ZERO,
                ))
            }
            _ => None,
        };
        instruction.map(TranspilerOutput::one_to_one)
    }
}
