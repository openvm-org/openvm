use openvm_decoder::{
    instruction_formats::{IType, RType},
    process_instruction,
};
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    riscv::REGISTER_NUM_LIMBS,
    LocalOpcode, PhantomDiscriminant, SystemOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_guest::{
    PhantomImm, ALU_OPCODE, ALU_OP_32, CSRRW_FUNCT3, CSR_OPCODE, HINT_BUFFER_IMM, HINT_FUNCT3,
    HINT_STORED_IMM, PHANTOM_FUNCT3, REVEAL_FUNCT3, RV64M_FUNCT7, SYSTEM_OPCODE, TERMINATE_FUNCT3,
};
pub use openvm_riscv_guest::{MAX_HINT_BUFFER_DWORDS, MAX_HINT_BUFFER_DWORDS_BITS};
use openvm_transpiler::{
    util::{nop, unimp},
    TranspilerExtension, TranspilerOutput,
};
use rrs::InstructionTranspiler;

mod instructions;
pub mod rrs;
pub use instructions::*;

#[derive(Default)]
pub struct Rv64ITranspilerExtension;

#[derive(Default)]
pub struct Rv64MTranspilerExtension;

#[derive(Default)]
pub struct Rv64IoTranspilerExtension;

impl TranspilerExtension for Rv64ITranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput> {
        let mut transpiler = InstructionTranspiler;
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
                    "Transpiling system / CSR instruction: {instruction_u32:b} (opcode = {opcode:07b}, funct3 = {funct3:03b}) to unimp"
                );
                return Some(TranspilerOutput::one_to_one(unimp()));
            }
            (SYSTEM_OPCODE, TERMINATE_FUNCT3) => {
                let dec_insn = IType::new(instruction_u32);
                let Ok(exit_code) = u8::try_from(dec_insn.imm) else {
                    return Some(TranspilerOutput::one_to_one(unimp()));
                };
                Some(Instruction {
                    opcode: SystemOpcode::TERMINATE.global_opcode(),
                    c: InstructionOperand::from(exit_code),
                    ..Default::default()
                })
            }
            (SYSTEM_OPCODE, PHANTOM_FUNCT3) => {
                let dec_insn = IType::new(instruction_u32);
                PhantomImm::from_repr(dec_insn.imm as u16).map(|phantom| match phantom {
                    PhantomImm::HintInput => Instruction::phantom(
                        PhantomDiscriminant(Rv64Phantom::HintInput as u16),
                        InstructionOperand::ZERO,
                        InstructionOperand::ZERO,
                        0,
                    ),
                    PhantomImm::HintRandom => Instruction::phantom(
                        PhantomDiscriminant(Rv64Phantom::HintRandom as u16),
                        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
                        InstructionOperand::ZERO,
                        0,
                    ),
                    PhantomImm::PrintStr => Instruction::phantom(
                        PhantomDiscriminant(Rv64Phantom::PrintStr as u16),
                        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
                        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
                        0,
                    ),
                })
            }
            (ALU_OPCODE | ALU_OP_32, _) => {
                // Exclude RV64M instructions from this transpiler extension
                let dec_insn = RType::new(instruction_u32);
                let funct7 = dec_insn.funct7 as u8;
                match funct7 {
                    RV64M_FUNCT7 => None,
                    _ => process_instruction(&mut transpiler, instruction_u32),
                }
            }
            _ => process_instruction(&mut transpiler, instruction_u32),
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl TranspilerExtension for Rv64MTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        if opcode != ALU_OPCODE && opcode != ALU_OP_32 {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let funct7 = dec_insn.funct7 as u8;
        if funct7 != RV64M_FUNCT7 {
            return None;
        }

        let instruction = process_instruction(&mut InstructionTranspiler, instruction_u32);

        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl TranspilerExtension for Rv64IoTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != SYSTEM_OPCODE {
            return None;
        }

        let instruction = match funct3 {
            HINT_FUNCT3 => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                match imm_u16 {
                    HINT_STORED_IMM => Some(Instruction::from_isize(
                        HintStoreOpcode::HINT_STORED.global_opcode(),
                        0,
                        (REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                        0,
                        1,
                        2,
                    )),
                    HINT_BUFFER_IMM => Some(Instruction::from_isize(
                        HintStoreOpcode::HINT_BUFFER.global_opcode(),
                        (REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                        (REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
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
                Some(Instruction::large_from_isize(
                    RevealOpcode::REVEAL.global_opcode(),
                    (REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    PUBLIC_VALUES_AS as isize,
                    1,
                    (dec_insn.imm < 0) as isize,
                ))
            }
            _ => return None,
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{
        instruction::Instruction, riscv::REGISTER_NUM_LIMBS, LocalOpcode, PUBLIC_VALUES_AS,
    };
    use openvm_riscv_guest::{ALU_OPCODE, REVEAL_FUNCT3, SYSTEM_OPCODE, TERMINATE_FUNCT3};
    use openvm_transpiler::{util::unimp, TranspilerExtension};

    use super::{RevealOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension};

    fn encode_reveal(rs1: u32, rd: u32, imm: i32) -> u32 {
        debug_assert!((-(1 << 11)..(1 << 11)).contains(&imm));
        ((imm as u32 & 0xfff) << 20)
            | (rs1 << 15)
            | (u32::from(REVEAL_FUNCT3) << 12)
            | (rd << 7)
            | u32::from(SYSTEM_OPCODE)
    }

    fn transpile(instruction: u32) -> Option<Instruction> {
        Rv64IoTranspilerExtension
            .process_custom(&[instruction])?
            .instructions
            .into_iter()
            .next()?
    }

    #[test]
    fn reveal_preserves_legacy_operands() {
        for (rs1, rd, imm) in [(7, 3, 123), (31, 0, -2048), (0, 31, 2047)] {
            let actual =
                transpile(encode_reveal(rs1, rd, imm)).expect("well-formed REVEAL must transpile");
            let expected = Instruction::large_from_isize(
                RevealOpcode::REVEAL.global_opcode(),
                (REGISTER_NUM_LIMBS * rs1 as usize) as isize,
                (REGISTER_NUM_LIMBS * rd as usize) as isize,
                (imm as i16 as u16) as isize,
                1,
                PUBLIC_VALUES_AS as isize,
                1,
                isize::from(imm.is_negative()),
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn reveal_rejects_non_reveal_instruction_shapes() {
        let reveal = encode_reveal(7, 3, -1);
        let wrong_opcode = (reveal & !0x7f) | u32::from(ALU_OPCODE);
        let wrong_funct3 = (reveal & !(0b111 << 12)) | (0b111 << 12);

        assert!(transpile(wrong_opcode).is_none());
        assert!(transpile(wrong_funct3).is_none());
        assert!(Rv64IoTranspilerExtension.process_custom(&[]).is_none());
    }

    fn terminate_instruction(exit_code: u32) -> u32 {
        (exit_code << 20) | (u32::from(TERMINATE_FUNCT3) << 12) | u32::from(SYSTEM_OPCODE)
    }

    #[test]
    fn terminate_accepts_byte_exit_code() {
        let output = Rv64ITranspilerExtension
            .process_custom(&[terminate_instruction(u8::MAX.into())])
            .unwrap();
        let instruction = output.instructions[0].as_ref().unwrap();

        assert_eq!(instruction.c.as_u32(), u32::from(u8::MAX));
    }

    #[test]
    fn terminate_maps_non_byte_exit_code_to_unimp() {
        let output = Rv64ITranspilerExtension
            .process_custom(&[terminate_instruction(u8::MAX as u32 + 1)])
            .unwrap();
        let instruction = output.instructions[0].as_ref().unwrap();

        assert_eq!(instruction, &unimp());
    }
}
