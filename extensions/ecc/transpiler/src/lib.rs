use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, Rv32WeierstrassOpcode, UsizeOpcode,
};
use axvm_transpiler::{util::from_r_type, TranspilerExtension};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::RType;
use strum::EnumCount;
use strum_macros::FromRepr;

#[derive(Default)]
pub struct EccTranspilerExtension;

// TODO: the opcode and func3 will be imported from `guest` crate
pub(crate) const OPCODE: u8 = 0x2b;
pub(crate) const FUNCT3: u8 = 0b001;

// TODO: this should be moved to `guest` crate
pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;

/// Short Weierstrass curves are configurable.
/// The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    SwAddNe = 0,
    SwDouble,
    SwSetup,
}

impl<F: PrimeField32> TranspilerExtension<F> for EccTranspilerExtension {
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
        if funct3 != FUNCT3 {
            return None;
        }

        let instruction = {
            // short weierstrass ec
            assert!(Rv32WeierstrassOpcode::COUNT <= SHORT_WEIERSTRASS_MAX_KINDS as usize);
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % SHORT_WEIERSTRASS_MAX_KINDS;
            let curve_idx_shift = ((dec_insn.funct7 as u8) / SHORT_WEIERSTRASS_MAX_KINDS) as usize
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
        };
        instruction.map(|instruction| (instruction, 1))
    }
}
