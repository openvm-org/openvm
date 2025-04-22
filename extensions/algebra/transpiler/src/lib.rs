use openvm_algebra_guest::{
    ComplexExtFieldBaseFunct7, ModArithBaseFunct7, COMPLEX_EXT_FIELD_FUNCT3,
    MODULAR_ARITHMETIC_FUNCT3, OPCODE,
};
use openvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode, PhantomDiscriminant,
    VmOpcode,
};
use openvm_instructions_derive::LocalOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::RType;
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum ModularPhantom {
    HintNonQr = 0x50,
    HintSqrt = 0x51,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
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
            let mod_idx = ((dec_insn.funct7 as u8)
                / ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS)
                as usize;
            let mod_idx_shift = mod_idx * Rv32ModularArithmeticOpcode::COUNT;
            if base_funct7 == ModArithBaseFunct7::SetupMod as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Rv32ModularArithmeticOpcode::SETUP_ADDSUB,
                    1 => Rv32ModularArithmeticOpcode::SETUP_MULDIV,
                    2 => Rv32ModularArithmeticOpcode::SETUP_ISEQ,
                    _ => panic!("invalid opcode"),
                };
                if local_opcode == Rv32ModularArithmeticOpcode::SETUP_ISEQ && dec_insn.rd == 0 {
                    panic!("SETUP_ISEQ is not valid for rd = x0");
                } else {
                    Some(Instruction::new(
                        VmOpcode::from_usize(
                            local_opcode.global_opcode().as_usize() + mod_idx_shift,
                        ),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                        F::ZERO, // rs2 = 0
                        F::ONE,  // d_as = 1
                        F::TWO,  // e_as = 2
                        F::ZERO,
                        F::ZERO,
                    ))
                }
            } else if base_funct7 == ModArithBaseFunct7::HintNonQr as u8 {
                assert_eq!(dec_insn.rd, 0);
                assert_eq!(dec_insn.rs1, 0);
                assert_eq!(dec_insn.rs2, 0);
                Some(Instruction::phantom(
                    PhantomDiscriminant(ModularPhantom::HintNonQr as u16),
                    F::ZERO,
                    F::ZERO,
                    mod_idx as u16,
                ))
            } else if base_funct7 == ModArithBaseFunct7::HintSqrt as u8 {
                assert_eq!(dec_insn.rd, 0);
                assert_eq!(dec_insn.rs2, 0);
                Some(Instruction::phantom(
                    PhantomDiscriminant(ModularPhantom::HintSqrt as u16),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::ZERO,
                    mod_idx as u16,
                ))
            } else {
                let global_opcode = match ModArithBaseFunct7::from_repr(base_funct7) {
                    Some(ModArithBaseFunct7::AddMod) => {
                        Rv32ModularArithmeticOpcode::ADD as usize
                            + Rv32ModularArithmeticOpcode::CLASS_OFFSET
                    }
                    Some(ModArithBaseFunct7::SubMod) => {
                        Rv32ModularArithmeticOpcode::SUB as usize
                            + Rv32ModularArithmeticOpcode::CLASS_OFFSET
                    }
                    Some(ModArithBaseFunct7::MulMod) => {
                        Rv32ModularArithmeticOpcode::MUL as usize
                            + Rv32ModularArithmeticOpcode::CLASS_OFFSET
                    }
                    Some(ModArithBaseFunct7::DivMod) => {
                        Rv32ModularArithmeticOpcode::DIV as usize
                            + Rv32ModularArithmeticOpcode::CLASS_OFFSET
                    }
                    Some(ModArithBaseFunct7::IsEqMod) => {
                        Rv32ModularArithmeticOpcode::IS_EQ as usize
                            + Rv32ModularArithmeticOpcode::CLASS_OFFSET
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode + mod_idx_shift;
                // The only opcode in this extension which can write to rd is `IsEqMod`
                // so we cannot allow rd to be zero in this case.
                let allow_rd_zero =
                    ModArithBaseFunct7::from_repr(base_funct7) != Some(ModArithBaseFunct7::IsEqMod);
                Some(from_r_type(global_opcode, 2, &dec_insn, allow_rd_zero))
            }
        };
        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Fp2TranspilerExtension {
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
                    VmOpcode::from_usize(
                        local_opcode.global_opcode().as_usize() + complex_idx_shift,
                    ),
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
                        Fp2Opcode::ADD as usize + Fp2Opcode::CLASS_OFFSET
                    }
                    Some(ComplexExtFieldBaseFunct7::Sub) => {
                        Fp2Opcode::SUB as usize + Fp2Opcode::CLASS_OFFSET
                    }
                    Some(ComplexExtFieldBaseFunct7::Mul) => {
                        Fp2Opcode::MUL as usize + Fp2Opcode::CLASS_OFFSET
                    }
                    Some(ComplexExtFieldBaseFunct7::Div) => {
                        Fp2Opcode::DIV as usize + Fp2Opcode::CLASS_OFFSET
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode + complex_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn, true))
            }
        };
        instruction.map(TranspilerOutput::one_to_one)
    }
}
