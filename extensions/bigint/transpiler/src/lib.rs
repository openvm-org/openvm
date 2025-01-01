use openvm_bigint_guest::{Int256Funct7, BEQ256_FUNCT3, INT256_FUNCT3, OPCODE};
use openvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, utils::isize_to_field, UsizeOpcode,
    VmOpcode,
};
use openvm_instructions_derive::UsizeOpcode;
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::{BType, RType};
use strum::IntoEnumIterator;

// =================================================================================================
// Intrinsics: 256-bit Integers
// =================================================================================================

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x400]
pub struct Rv32BaseAlu256Opcode(pub BaseAluOpcode);

impl Rv32BaseAlu256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BaseAluOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x405]
pub struct Rv32Shift256Opcode(pub ShiftOpcode);

impl Rv32Shift256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        ShiftOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x408]
pub struct Rv32LessThan256Opcode(pub LessThanOpcode);

impl Rv32LessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        LessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x420]
pub struct Rv32BranchEqual256Opcode(pub BranchEqualOpcode);

impl Rv32BranchEqual256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x425]
pub struct Rv32BranchLessThan256Opcode(pub BranchLessThanOpcode);

impl Rv32BranchLessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchLessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x450]
pub struct Rv32Mul256Opcode(pub MulOpcode);

impl Rv32Mul256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        MulOpcode::iter().map(Self)
    }
}

#[derive(Default)]
pub struct Int256TranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Int256TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
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
                    VmOpcode::from_usize(
                        BranchEqualOpcode::BEQ as usize
                            + Rv32BranchEqual256Opcode::default_offset(),
                    ),
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
        instruction.map(TranspilerOutput::one_to_one)
    }
}
