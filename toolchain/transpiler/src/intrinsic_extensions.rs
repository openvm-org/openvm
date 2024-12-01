use axvm_instructions::*;
use axvm_platform::constants::{Custom0Funct3::*, *};
use instruction::Instruction;
use p3_field::PrimeField32;
use riscv::RV32_REGISTER_NUM_LIMBS;
use rrs_lib::instruction_formats::IType;

use crate::{
    util::{nop, terminate, unimp},
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
        CUSTOM_1 => None,
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
