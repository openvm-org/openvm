use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, Fp2Opcode,
    Rv32ModularArithmeticOpcode, UsizeOpcode,
};
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::RType;
use strum::EnumCount;
use strum_macros::FromRepr;

#[derive(Default)]
pub struct ModularTranspilerExtension;

#[derive(Default)]
pub struct Fp2TranspilerExtension;

// TODO: the opcode and func3 will be imported from `guest` crate
pub(crate) const OPCODE: u8 = 0x2b;
pub(crate) const MODULAR_ARITHMETIC_FUNCT3: u8 = 0b000;
pub(crate) const COMPLEX_EXT_FIELD_FUNCT3: u8 = 0b010;

// TODO: this should be moved to `guest` crate
pub const MODULAR_ARITHMETIC_MAX_KINDS: u8 = 8;

/// Modular arithmetic is configurable.
/// The funct7 field equals `mod_idx * MODULAR_ARITHMETIC_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ModArithBaseFunct7 {
    AddMod = 0,
    SubMod,
    MulMod,
    DivMod,
    IsEqMod,
    SetupMod,
}

pub const COMPLEX_EXT_FIELD_MAX_KINDS: u8 = 8;

/// Complex extension field is configurable.
/// The funct7 field equals `fp2_idx * COMPLEX_EXT_FIELD_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ComplexExtFieldBaseFunct7 {
    Add = 0,
    Sub,
    Mul,
    Div,
    Setup,
}

impl<F: PrimeField32> TranspilerExtension<F> for ModularTranspilerExtension {
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
        if funct3 != MODULAR_ARITHMETIC_FUNCT3 {
            return None;
        }

        let instruction = {
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % MODULAR_ARITHMETIC_MAX_KINDS;
            assert!(Rv32ModularArithmeticOpcode::COUNT <= MODULAR_ARITHMETIC_MAX_KINDS as usize);
            let mod_idx_shift = ((dec_insn.funct7 as u8) / MODULAR_ARITHMETIC_MAX_KINDS) as usize
                * Rv32ModularArithmeticOpcode::COUNT;
            if base_funct7 == ModArithBaseFunct7::SetupMod as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Rv32ModularArithmeticOpcode::SETUP_ADDSUB,
                    1 => Rv32ModularArithmeticOpcode::SETUP_MULDIV,
                    2 => Rv32ModularArithmeticOpcode::SETUP_ISEQ,
                    _ => panic!("invalid opcode"),
                };
                Some(Instruction::new(
                    local_opcode.with_default_offset() + mod_idx_shift,
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::ZERO, // rs2 = 0
                    F::ONE,  // d_as = 1
                    F::TWO,  // e_as = 2
                    F::ZERO,
                    F::ZERO,
                ))
            } else {
                let global_opcode = match ModArithBaseFunct7::from_repr(base_funct7) {
                    Some(ModArithBaseFunct7::AddMod) => {
                        Rv32ModularArithmeticOpcode::ADD as usize
                            + Rv32ModularArithmeticOpcode::default_offset()
                    }
                    Some(ModArithBaseFunct7::SubMod) => {
                        Rv32ModularArithmeticOpcode::SUB as usize
                            + Rv32ModularArithmeticOpcode::default_offset()
                    }
                    Some(ModArithBaseFunct7::MulMod) => {
                        Rv32ModularArithmeticOpcode::MUL as usize
                            + Rv32ModularArithmeticOpcode::default_offset()
                    }
                    Some(ModArithBaseFunct7::DivMod) => {
                        Rv32ModularArithmeticOpcode::DIV as usize
                            + Rv32ModularArithmeticOpcode::default_offset()
                    }
                    Some(ModArithBaseFunct7::IsEqMod) => {
                        Rv32ModularArithmeticOpcode::IS_EQ as usize
                            + Rv32ModularArithmeticOpcode::default_offset()
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode + mod_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn))
            }
        };
        instruction.map(|instruction| (instruction, 1))
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Fp2TranspilerExtension {
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
        if funct3 != COMPLEX_EXT_FIELD_FUNCT3 {
            return None;
        }

        let instruction = {
            assert!(Fp2Opcode::COUNT <= COMPLEX_EXT_FIELD_MAX_KINDS as usize);
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % COMPLEX_EXT_FIELD_MAX_KINDS;
            let complex_idx_shift =
                ((dec_insn.funct7 as u8) / COMPLEX_EXT_FIELD_MAX_KINDS) as usize * Fp2Opcode::COUNT;

            if base_funct7 == ComplexExtFieldBaseFunct7::Setup as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Fp2Opcode::SETUP_ADDSUB,
                    1 => Fp2Opcode::SETUP_MULDIV,
                    _ => panic!("invalid opcode"),
                };
                Some(Instruction::new(
                    local_opcode.with_default_offset() + complex_idx_shift,
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::ZERO, // rs2 = 0
                    F::ONE,  // d_as = 1
                    F::TWO,  // e_as = 2
                    F::ZERO,
                    F::ZERO,
                ))
            } else {
                let global_opcode = match ComplexExtFieldBaseFunct7::from_repr(base_funct7) {
                    Some(ComplexExtFieldBaseFunct7::Add) => {
                        Fp2Opcode::ADD as usize + Fp2Opcode::default_offset()
                    }
                    Some(ComplexExtFieldBaseFunct7::Sub) => {
                        Fp2Opcode::SUB as usize + Fp2Opcode::default_offset()
                    }
                    Some(ComplexExtFieldBaseFunct7::Mul) => {
                        Fp2Opcode::MUL as usize + Fp2Opcode::default_offset()
                    }
                    Some(ComplexExtFieldBaseFunct7::Div) => {
                        Fp2Opcode::DIV as usize + Fp2Opcode::default_offset()
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode + complex_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn))
            }
        };
        instruction.map(|instruction| (instruction, 1))
    }
}
