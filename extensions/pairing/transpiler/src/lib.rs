use openvm_decoder::instruction_formats::RType;
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    riscv::REGISTER_NUM_LIMBS,
    PhantomDiscriminant,
};
use openvm_pairing_guest::{PairingBaseFunct7, OPCODE, PAIRING_FUNCT3};
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};
use strum::FromRepr;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum PairingPhantom {
    /// Uses `b` to determine the curve: `b` is the discriminant of `PairingCurve` kind.
    /// Peeks at `[r32{0}(a)..r32{0}(a) + Fp::NUM_LIMBS * 12]_2` to get `f: Fp12` and then resets
    /// the hint stream to equal `final_exp_hint(f) = (residue_witness, scaling_factor): (Fp12,
    /// Fp12)` as `Fp::NUM_LIMBS * 12 * 2` bytes.
    HintFinalExp = 0x30,
}

#[derive(Default)]
pub struct PairingTranspilerExtension;

impl TranspilerExtension for PairingTranspilerExtension {
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
        if funct3 != PAIRING_FUNCT3 {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let base_funct7 = (dec_insn.funct7 as u8) % PairingBaseFunct7::PAIRING_MAX_KINDS;
        let pairing_idx = ((dec_insn.funct7 as u8) / PairingBaseFunct7::PAIRING_MAX_KINDS) as usize;
        if let Some(PairingBaseFunct7::HintFinalExp) = PairingBaseFunct7::from_repr(base_funct7) {
            assert_eq!(dec_insn.rd, 0);
            // Return exits the outermost function
            return Some(TranspilerOutput::one_to_one(Instruction::phantom(
                PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
                InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
                InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
                pairing_idx as u16,
            )));
        }
        None
    }
}
