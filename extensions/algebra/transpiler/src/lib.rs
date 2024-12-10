use ax_stark_backend::p3_field::PrimeField32;
use axvm_algebra_guest::{
    ComplexExtFieldBaseFunct7, ModArithBaseFunct7, COMPLEX_EXT_FIELD_FUNCT3,
    MODULAR_ARITHMETIC_FUNCT3, OPCODE,
};
use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, AxVmOpcode, UsizeOpcode,
};
use axvm_instructions_derive::UsizeOpcode;
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x500]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32ModularArithmeticOpcode {
    ADD,
    SUB,
    SETUP_ADDSUB,
    MUL,
    DIV,
    SETUP_MULDIV,
    IS_EQ,
    SETUP_ISEQ,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x710]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Fp2Opcode {
    ADD,
    SUB,
    SETUP_ADDSUB,
    MUL,
    DIV,
    SETUP_MULDIV,
}

#[derive(Default)]
pub struct ModularTranspilerExtension;

#[derive(Default)]
pub struct Fp2TranspilerExtension;

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
            let base_funct7 =
                (dec_insn.funct7 as u8) % ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS;
            assert!(
                Rv32ModularArithmeticOpcode::COUNT
                    <= ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize
            );
            let mod_idx_shift = ((dec_insn.funct7 as u8)
                / ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS)
                as usize
                * Rv32ModularArithmeticOpcode::COUNT;
            if base_funct7 == ModArithBaseFunct7::SetupMod as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Rv32ModularArithmeticOpcode::SETUP_ADDSUB,
                    1 => Rv32ModularArithmeticOpcode::SETUP_MULDIV,
                    2 => Rv32ModularArithmeticOpcode::SETUP_ISEQ,
                    _ => panic!("invalid opcode"),
                };
                Some(Instruction::new(
                    AxVmOpcode::from_usize(local_opcode.with_default_offset() + mod_idx_shift),
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
            assert!(
                Fp2Opcode::COUNT <= ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS as usize
            );
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 =
                (dec_insn.funct7 as u8) % ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS;
            let complex_idx_shift = ((dec_insn.funct7 as u8)
                / ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS)
                as usize
                * Fp2Opcode::COUNT;

            if base_funct7 == ComplexExtFieldBaseFunct7::Setup as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Fp2Opcode::SETUP_ADDSUB,
                    1 => Fp2Opcode::SETUP_MULDIV,
                    _ => panic!("invalid opcode"),
                };
                Some(Instruction::new(
                    AxVmOpcode::from_usize(local_opcode.with_default_offset() + complex_idx_shift),
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