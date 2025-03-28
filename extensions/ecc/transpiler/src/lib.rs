use openvm_ecc_guest::{SwBaseFunct7, TeBaseFunct7, SW_FUNCT3, SW_OPCODE, TE_FUNCT3, TE_OPCODE};
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
#[opcode_offset = 0x600]
#[allow(non_camel_case_types)]
#[repr(usize)]
pub enum Rv32WeierstrassOpcode {
    SW_ADD_NE,
    SETUP_SW_ADD_NE,
    SW_DOUBLE,
    SETUP_SW_DOUBLE,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x680]
#[allow(non_camel_case_types)]
#[repr(usize)]
pub enum Rv32EdwardsOpcode {
    TE_ADD,
    SETUP_TE_ADD,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum EccPhantom {
    SwHintDecompress = 0x40,
    TeHintDecompress = 0x41,
    HintNonQr = 0x42,
}

#[derive(Default)]
pub struct EccTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for EccTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        self.process_weierstrass_instruction(instruction_stream)
            .or(self.process_edwards_instruction(instruction_stream))
    }
}

impl EccTranspilerExtension {
    fn process_edwards_instruction<F: PrimeField32>(
        &self,
        instruction_stream: &[u32],
    ) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != TE_OPCODE {
            return None;
        }
        if funct3 != TE_FUNCT3 {
            return None;
        }

        let instruction = {
            // twisted edwards ec
            assert!(Rv32EdwardsOpcode::COUNT <= TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize);
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS;
            let curve_idx =
                ((dec_insn.funct7 as u8) / TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS) as usize;
            let curve_idx_shift = curve_idx * Rv32EdwardsOpcode::COUNT;

            if let Some(TeBaseFunct7::TeHintDecompress) = TeBaseFunct7::from_repr(base_funct7) {
                assert_eq!(dec_insn.rd, 0);
                return Some(TranspilerOutput::one_to_one(Instruction::phantom(
                    PhantomDiscriminant(EccPhantom::TeHintDecompress as u16),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    curve_idx as u16,
                )));
            }
            if base_funct7 == TeBaseFunct7::TeSetup as u8 {
                let local_opcode = Rv32EdwardsOpcode::SETUP_TE_ADD;
                Some(Instruction::new(
                    VmOpcode::from_usize(local_opcode.global_opcode().as_usize() + curve_idx_shift),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    F::ONE, // d_as = 1
                    F::TWO, // e_as = 2
                    F::ZERO,
                    F::ZERO,
                ))
            } else {
                let global_opcode = match TeBaseFunct7::from_repr(base_funct7) {
                    Some(TeBaseFunct7::TeAdd) => Rv32EdwardsOpcode::TE_ADD.global_opcode(),
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode.as_usize() + curve_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn, true))
            }
        };
        instruction.map(TranspilerOutput::one_to_one)
    }

    fn process_weierstrass_instruction<F: PrimeField32>(
        &self,
        instruction_stream: &[u32],
    ) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != SW_OPCODE {
            return None;
        }
        if funct3 != SW_FUNCT3 {
            return None;
        }

        let instruction = {
            // short weierstrass ec
            assert!(
                Rv32WeierstrassOpcode::COUNT <= SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize
            );
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS;
            let curve_idx =
                ((dec_insn.funct7 as u8) / SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS) as usize;
            let curve_idx_shift = curve_idx * Rv32WeierstrassOpcode::COUNT;

            if let Some(SwBaseFunct7::SwHintDecompress) = SwBaseFunct7::from_repr(base_funct7) {
                assert_eq!(dec_insn.rd, 0);
                return Some(TranspilerOutput::one_to_one(Instruction::phantom(
                    PhantomDiscriminant(EccPhantom::SwHintDecompress as u16),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    curve_idx as u16,
                )));
            }
            if let Some(SwBaseFunct7::HintNonQr) = SwBaseFunct7::from_repr(base_funct7) {
                assert_eq!(dec_insn.rd, 0);
                assert_eq!(dec_insn.rs1, 0);
                assert_eq!(dec_insn.rs2, 0);
                return Some(TranspilerOutput::one_to_one(Instruction::phantom(
                    PhantomDiscriminant(EccPhantom::HintNonQr as u16),
                    F::ZERO,
                    F::ZERO,
                    curve_idx as u16,
                )));
            }
            if base_funct7 == SwBaseFunct7::SwSetup as u8 {
                let local_opcode = match dec_insn.rs2 {
                    0 => Rv32WeierstrassOpcode::SETUP_SW_DOUBLE,
                    _ => Rv32WeierstrassOpcode::SETUP_SW_ADD_NE,
                };
                Some(Instruction::new(
                    VmOpcode::from_usize(local_opcode.global_opcode().as_usize() + curve_idx_shift),
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
                    Some(SwBaseFunct7::SwAddNe) => Rv32WeierstrassOpcode::SW_ADD_NE.global_opcode(),
                    Some(SwBaseFunct7::SwDouble) => {
                        assert!(dec_insn.rs2 == 0);
                        Rv32WeierstrassOpcode::SW_DOUBLE.global_opcode()
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode.as_usize() + curve_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn, true))
            }
        };
        instruction.map(TranspilerOutput::one_to_one)
    }
}
