use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, Fp12Opcode, PairingOpcode,
    PairingPhantom, PhantomDiscriminant, UsizeOpcode,
};
use axvm_pairing_guest::{PairingBaseFunct7, OPCODE, PAIRING_FUNCT3};
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::RType;
use strum::EnumCount;

#[derive(Default)]
pub struct PairingTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for PairingTranspilerExtension {
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
        if funct3 != PAIRING_FUNCT3 {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let base_funct7 = (dec_insn.funct7 as u8) % PairingBaseFunct7::PAIRING_MAX_KINDS;
        let pairing_idx = ((dec_insn.funct7 as u8) / PairingBaseFunct7::PAIRING_MAX_KINDS) as usize;
        if let Some(PairingBaseFunct7::HintFinalExp) = PairingBaseFunct7::from_repr(base_funct7) {
            assert_eq!(dec_insn.rd, 0);
            // Return exits the outermost function
            return Some((
                Instruction::phantom(
                    PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    pairing_idx as u16,
                ),
                1,
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
            Some(PairingBaseFunct7::Fp12Mul) => {
                Fp12Opcode::MUL as usize + Fp12Opcode::default_offset()
            }
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

        assert!(PairingOpcode::COUNT < PairingBaseFunct7::PAIRING_MAX_KINDS as usize); // + 1 for Fp12Mul
        let pairing_idx_shift =
            if let Some(PairingBaseFunct7::Fp12Mul) = PairingBaseFunct7::from_repr(base_funct7) {
                // SPECIAL CASE: Fp12Mul uses different enum Fp12Opcode
                pairing_idx * Fp12Opcode::COUNT
            } else {
                pairing_idx * PairingOpcode::COUNT
            };
        let global_opcode = global_opcode + pairing_idx_shift;

        Some((from_r_type(global_opcode, 2, &dec_insn), 1))
    }
}
