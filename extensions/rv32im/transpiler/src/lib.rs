use std::marker::PhantomData;

use openvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode, PhantomDiscriminant,
    SystemOpcode,
};
use openvm_rv32im_guest::{
    PhantomImm, CSRRW_FUNCT3, CSR_OPCODE, HINT_BUFFER_IMM, HINT_FUNCT3, HINT_STOREW_IMM,
    NATIVE_STOREW_FUNCT3, NATIVE_STOREW_FUNCT7, PHANTOM_FUNCT3, REVEAL_FUNCT3, RV32M_FUNCT7,
    RV32_ALU_OPCODE, SYSTEM_OPCODE, TERMINATE_FUNCT3,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{
    util::{nop, unimp},
    TranspilerExtension, TranspilerOutput,
};
use rrs::InstructionTranspiler;
use rrs_lib::{
    instruction_formats::{IType, RType},
    process_instruction,
};

mod instructions;
pub mod rrs;
pub use instructions::*;

#[derive(Default)]
pub struct Rv32ITranspilerExtension;

#[derive(Default)]
pub struct Rv32MTranspilerExtension;

#[derive(Default)]
pub struct Rv32IoTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Rv32ITranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        let mut transpiler = InstructionTranspiler::<F>(PhantomData);
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R-, I- or B-type

        let instruction = match (opcode, funct3) {
            (CSR_OPCODE, _) => {
                let dec_insn = IType::new(instruction_u32);
                if dec_insn.funct3 as u8 == CSRRW_FUNCT3 {
                    // CSRRW
                    if dec_insn.rs1 == 0 && dec_insn.rd == 0 {
                        // This resets the CSR counter to zero. Since we don't have any CSR
                        // registers, this is a nop.
                        return Some(TranspilerOutput::one_to_one(nop()));
                    }
                }
                eprintln!(
                    "Transpiling system / CSR instruction: {:b} (opcode = {:07b}, funct3 = {:03b}) to unimp",
                    instruction_u32, opcode, funct3
                );
                return Some(TranspilerOutput::one_to_one(unimp()));
            }
            (SYSTEM_OPCODE, TERMINATE_FUNCT3) => {
                let dec_insn = IType::new(instruction_u32);
                Some(Instruction {
                    opcode: SystemOpcode::TERMINATE.global_opcode(),
                    c: F::from_canonical_u8(
                        dec_insn.imm.try_into().expect("exit code must be byte"),
                    ),
                    ..Default::default()
                })
            }
            (SYSTEM_OPCODE, PHANTOM_FUNCT3) => {
                let dec_insn = IType::new(instruction_u32);
                PhantomImm::from_repr(dec_insn.imm as u16).map(|phantom| match phantom {
                    PhantomImm::HintInput => Instruction::phantom(
                        PhantomDiscriminant(Rv32Phantom::HintInput as u16),
                        F::ZERO,
                        F::ZERO,
                        0,
                    ),
                    PhantomImm::HintRandom => Instruction::phantom(
                        PhantomDiscriminant(Rv32Phantom::HintRandom as u16),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                        F::ZERO,
                        0,
                    ),
                    PhantomImm::PrintStr => Instruction::phantom(
                        PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                        0,
                    ),
                    PhantomImm::HintLoadByKey => Instruction::phantom(
                        PhantomDiscriminant(Rv32Phantom::HintLoadByKey as u16),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                        0,
                    ),
                })
            }
            (RV32_ALU_OPCODE, _) => {
                // Exclude RV32M instructions from this transpiler extension
                let dec_insn = RType::new(instruction_u32);
                let funct7 = dec_insn.funct7 as u8;
                match funct7 {
                    RV32M_FUNCT7 => None,
                    _ => process_instruction(&mut transpiler, instruction_u32),
                }
            }
            _ => process_instruction(&mut transpiler, instruction_u32),
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv32MTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        if opcode != RV32_ALU_OPCODE {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let funct7 = dec_insn.funct7 as u8;
        if funct7 != RV32M_FUNCT7 {
            return None;
        }

        let instruction = process_instruction(
            &mut InstructionTranspiler::<F>(PhantomData),
            instruction_u32,
        );

        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv32IoTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R-, I- or B-type

        if opcode != SYSTEM_OPCODE {
            return None;
        }

        let instruction = match funct3 {
            HINT_FUNCT3 => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                match imm_u16 {
                    HINT_STOREW_IMM => Some(Instruction::from_isize(
                        Rv32HintStoreOpcode::HINT_STOREW.global_opcode(),
                        0,
                        (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                        0,
                        1,
                        2,
                    )),
                    HINT_BUFFER_IMM => Some(Instruction::from_isize(
                        Rv32HintStoreOpcode::HINT_BUFFER.global_opcode(),
                        (RV32_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                        (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                        0,
                        1,
                        2,
                    )),
                    _ => None,
                }
            }
            REVEAL_FUNCT3 => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                // REVEAL_RV32 is a pseudo-instruction for STOREW_RV32 a,b,c,1,3
                Some(Instruction::large_from_isize(
                    Rv32LoadStoreOpcode::STOREW.global_opcode(),
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    3,
                    1,
                    (dec_insn.imm < 0) as isize,
                ))
            }
            NATIVE_STOREW_FUNCT3 => {
                // NATIVE_STOREW is a pseudo-instruction for STOREW_RV32 a,b,0,1,4
                let dec_insn = RType::new(instruction_u32);
                if dec_insn.funct7 != NATIVE_STOREW_FUNCT7 {
                    return None;
                }
                Some(Instruction::large_from_isize(
                    Rv32LoadStoreOpcode::STOREW.global_opcode(),
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    0,
                    1,
                    4,
                    1,
                    0,
                ))
            }
            _ => return None,
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}
