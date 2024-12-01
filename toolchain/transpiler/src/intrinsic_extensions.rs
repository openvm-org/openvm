use axvm_instructions::*;
use axvm_platform::constants::{Custom0Funct3::*, Custom1Funct3::*, *};
use instruction::Instruction;
use p3_field::PrimeField32;
use riscv::RV32_REGISTER_NUM_LIMBS;
use rrs_lib::instruction_formats::{IType, RType};
use strum::EnumCount;

use crate::{
    util::{from_r_type, nop, terminate, unimp},
    TranspilerExtension,
};

fn process_custom_instruction<F: PrimeField32>(instruction_u32: u32) -> Option<Instruction<F>> {
    let opcode = (instruction_u32 & 0x7f) as u8;
    let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R-, I- or B-type

    let result = match opcode {
        CUSTOM_0 => match Custom0Funct3::from_repr(funct3) {
            Some(Terminate) => {
                let dec_insn = IType::new(instruction_u32);
                Some(terminate(
                    dec_insn.imm.try_into().expect("exit code must be byte"),
                ))
            }
            Some(HintStoreW) => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                Some(Instruction::from_isize(
                    Rv32HintStoreOpcode::HINT_STOREW.with_default_offset(),
                    0,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    2,
                ))
            }
            Some(Reveal) => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                // REVEAL_RV32 is a pseudo-instruction for STOREW_RV32 a,b,c,1,3
                Some(Instruction::from_isize(
                    Rv32LoadStoreOpcode::STOREW.with_default_offset(),
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    3,
                ))
            }
            Some(Phantom) => process_phantom(instruction_u32),

            _ => None,
        },
        CUSTOM_1 => {
            match Custom1Funct3::from_repr(funct3) {
                Some(ShortWeierstrass) => {
                    // short weierstrass ec
                    assert!(Rv32WeierstrassOpcode::COUNT <= SHORT_WEIERSTRASS_MAX_KINDS as usize);
                    let dec_insn = RType::new(instruction_u32);
                    let base_funct7 = (dec_insn.funct7 as u8) % SHORT_WEIERSTRASS_MAX_KINDS;
                    let curve_idx_shift = ((dec_insn.funct7 as u8) / SHORT_WEIERSTRASS_MAX_KINDS)
                        as usize
                        * Rv32WeierstrassOpcode::COUNT;
                    if base_funct7 == SwBaseFunct7::SwSetup as u8 {
                        let local_opcode = match dec_insn.rs2 {
                            0 => Rv32WeierstrassOpcode::SETUP_EC_DOUBLE,
                            _ => Rv32WeierstrassOpcode::SETUP_EC_ADD_NE,
                        };
                        Some(Instruction::new(
                            local_opcode.with_default_offset() + curve_idx_shift,
                            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                            F::ONE, // d_as = 1
                            F::TWO, // e_as = 2
                            F::ZERO,
                            F::ZERO,
                        ))
                    } else {
                        let global_opcode = match SwBaseFunct7::from_repr(base_funct7) {
                            Some(SwBaseFunct7::SwAddNe) => {
                                Rv32WeierstrassOpcode::EC_ADD_NE as usize
                                    + Rv32WeierstrassOpcode::default_offset()
                            }
                            Some(SwBaseFunct7::SwDouble) => {
                                assert!(dec_insn.rs2 == 0);
                                Rv32WeierstrassOpcode::EC_DOUBLE as usize
                                    + Rv32WeierstrassOpcode::default_offset()
                            }
                            _ => unimplemented!(),
                        };
                        let global_opcode = global_opcode + curve_idx_shift;
                        Some(from_r_type(global_opcode, 2, &dec_insn))
                    }
                }
                Some(Pairing) => process_pairing(instruction_u32),
                _ => None,
            }
        }
        _ => None,
    };

    if result.is_some() {
        return result;
    }

    if opcode == 0b1110011 {
        let dec_insn = IType::new(instruction_u32);
        if dec_insn.funct3 == 0b001 {
            // CSRRW
            if dec_insn.rs1 == 0 && dec_insn.rd == 0 {
                // This resets the CSR counter to zero. Since we don't have any CSR registers, this is a nop.
                return Some(nop());
            }
        }
        eprintln!(
            "Transpiling system / CSR instruction: {:b} (opcode = {:07b}, funct3 = {:03b}) to unimp",
            instruction_u32, opcode, funct3
        );
        return Some(unimp());
    }

    None
}

fn process_phantom<F: PrimeField32>(instruction_u32: u32) -> Option<Instruction<F>> {
    let dec_insn = IType::new(instruction_u32);
    PhantomImm::from_repr(dec_insn.imm as u16).map(|phantom| match phantom {
        PhantomImm::HintInput => Instruction::phantom(
            PhantomDiscriminant(Rv32Phantom::HintInput as u16),
            F::ZERO,
            F::ZERO,
            0,
        ),
        PhantomImm::PrintStr => Instruction::phantom(
            PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            0,
        ),
    })
}

fn process_pairing<F: PrimeField32>(instruction_u32: u32) -> Option<Instruction<F>> {
    let dec_insn = RType::new(instruction_u32);
    let base_funct7 = (dec_insn.funct7 as u8) % PAIRING_MAX_KINDS;
    let pairing_idx = ((dec_insn.funct7 as u8) / PAIRING_MAX_KINDS) as usize;
    if let Some(PairingBaseFunct7::HintFinalExp) = PairingBaseFunct7::from_repr(base_funct7) {
        assert_eq!(dec_insn.rd, 0);
        // Return exits the outermost function
        return Some(Instruction::phantom(
            PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
            pairing_idx as u16,
        ));
    }
    let global_opcode = match PairingBaseFunct7::from_repr(base_funct7) {
        Some(PairingBaseFunct7::MillerDoubleStep) => {
            assert_eq!(dec_insn.rs2, 0);
            PairingOpcode::MILLER_DOUBLE_STEP as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::MillerDoubleAndAddStep) => {
            PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::Fp12Mul) => Fp12Opcode::MUL as usize + Fp12Opcode::default_offset(),
        Some(PairingBaseFunct7::EvaluateLine) => {
            PairingOpcode::EVALUATE_LINE as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::Mul013By013) => {
            PairingOpcode::MUL_013_BY_013 as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::MulBy01234) => {
            PairingOpcode::MUL_BY_01234 as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::Mul023By023) => {
            PairingOpcode::MUL_023_BY_023 as usize + PairingOpcode::default_offset()
        }
        Some(PairingBaseFunct7::MulBy02345) => {
            PairingOpcode::MUL_BY_02345 as usize + PairingOpcode::default_offset()
        }
        _ => unimplemented!(),
    };
    assert!(PairingOpcode::COUNT < PAIRING_MAX_KINDS as usize); // + 1 for Fp12Mul
    let pairing_idx_shift =
        if let Some(PairingBaseFunct7::Fp12Mul) = PairingBaseFunct7::from_repr(base_funct7) {
            // SPECIAL CASE: Fp12Mul uses different enum Fp12Opcode
            pairing_idx * Fp12Opcode::COUNT
        } else {
            pairing_idx * PairingOpcode::COUNT
        };
    let global_opcode = global_opcode + pairing_idx_shift;
    Some(from_r_type(global_opcode, 2, &dec_insn))
}

// TODO: rename and modularize this and move to separate crates
#[derive(Default)]
pub(crate) struct IntrinsicTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for IntrinsicTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let instruction = process_custom_instruction(instruction_u32);
        instruction.map(|ret| (ret, 1))
    }
}
