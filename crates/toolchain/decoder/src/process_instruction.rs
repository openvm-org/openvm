// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::instruction_formats;
use super::InstructionProcessor;

fn process_opcode_op<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::RType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_add(dec_insn)),
            0b000_0001 => Some(processor.process_mul(dec_insn)),
            0b010_0000 => Some(processor.process_sub(dec_insn)),
            _ => None,
        },
        0b001 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sll(dec_insn)),
            0b000_0001 => Some(processor.process_mulh(dec_insn)),
            _ => None,
        },
        0b010 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_slt(dec_insn)),
            0b000_0001 => Some(processor.process_mulhsu(dec_insn)),
            _ => None,
        },
        0b011 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sltu(dec_insn)),
            0b000_0001 => Some(processor.process_mulhu(dec_insn)),
            _ => None,
        },
        0b100 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_xor(dec_insn)),
            0b000_0001 => Some(processor.process_div(dec_insn)),
            _ => None,
        },
        0b101 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_srl(dec_insn)),
            0b000_0001 => Some(processor.process_divu(dec_insn)),
            0b010_0000 => Some(processor.process_sra(dec_insn)),
            _ => None,
        },
        0b110 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_or(dec_insn)),
            0b000_0001 => Some(processor.process_rem(dec_insn)),
            _ => None,
        },
        0b111 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_and(dec_insn)),
            0b000_0001 => Some(processor.process_remu(dec_insn)),
            _ => None,
        },
        _ => None,
    }
}

fn process_opcode_op_imm<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::IType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_addi(dec_insn)),
        0b001 => Some(processor.process_slli(instruction_formats::ITypeShamt::new(insn_bits))),
        0b010 => Some(processor.process_slti(dec_insn)),
        0b011 => Some(processor.process_sltui(dec_insn)),
        0b100 => Some(processor.process_xori(dec_insn)),
        0b101 => {
            let dec_insn_shamt = instruction_formats::ITypeShamt::new(insn_bits);
            match dec_insn_shamt.funct7 {
                0b000_0000 => Some(processor.process_srli(dec_insn_shamt)),
                0b010_0000 => Some(processor.process_srai(dec_insn_shamt)),
                _ => None,
            }
        }
        0b110 => Some(processor.process_ori(dec_insn)),
        0b111 => Some(processor.process_andi(dec_insn)),
        _ => None,
    }
}

fn process_opcode_branch<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::BType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_beq(dec_insn)),
        0b001 => Some(processor.process_bne(dec_insn)),
        0b100 => Some(processor.process_blt(dec_insn)),
        0b101 => Some(processor.process_bge(dec_insn)),
        0b110 => Some(processor.process_bltu(dec_insn)),
        0b111 => Some(processor.process_bgeu(dec_insn)),
        _ => None,
    }
}

fn process_opcode_load<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::IType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_lb(dec_insn)),
        0b001 => Some(processor.process_lh(dec_insn)),
        0b010 => Some(processor.process_lw(dec_insn)),
        0b100 => Some(processor.process_lbu(dec_insn)),
        0b101 => Some(processor.process_lhu(dec_insn)),
        _ => None,
    }
}

fn process_opcode_store<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::SType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_sb(dec_insn)),
        0b001 => Some(processor.process_sh(dec_insn)),
        0b010 => Some(processor.process_sw(dec_insn)),
        _ => None,
    }
}

fn process_opcode_system<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::ITypeCSR::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => {
            if dec_insn.rd != 0 || dec_insn.rs1 != 0 {
                None
            } else {
                match dec_insn.csr {
                    0b000000000000 => Some(processor.process_ecall()),
                    0b000000000001 => Some(processor.process_ebreak()),
                    0b000100000101 => Some(processor.process_wfi()),
                    0b001100000010 => Some(processor.process_mret()),
                    _ => None,
                }
            }
        }
        0b001 => Some(processor.process_csrrw(dec_insn)),
        0b010 => Some(processor.process_csrrs(dec_insn)),
        0b011 => Some(processor.process_csrrc(dec_insn)),
        0b101 => Some(processor.process_csrrwi(dec_insn)),
        0b110 => Some(processor.process_csrrsi(dec_insn)),
        0b111 => Some(processor.process_csrrci(dec_insn)),
        _ => None,
    }
}

/// Decodes instruction in `insn_bits` calling the appropriate function in `processor` returning
/// the result it produces.
///
/// Returns `None` if instruction doesn't decode into a valid instruction.
pub fn process_instruction<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let opcode: u32 = insn_bits & 0x7f;

    match opcode {
        instruction_formats::OPCODE_OP => process_opcode_op(processor, insn_bits),
        instruction_formats::OPCODE_OP_IMM => process_opcode_op_imm(processor, insn_bits),
        instruction_formats::OPCODE_LUI => {
            Some(processor.process_lui(instruction_formats::UType::new(insn_bits)))
        }
        instruction_formats::OPCODE_AUIPC => {
            Some(processor.process_auipc(instruction_formats::UType::new(insn_bits)))
        }
        instruction_formats::OPCODE_BRANCH => process_opcode_branch(processor, insn_bits),
        instruction_formats::OPCODE_LOAD => process_opcode_load(processor, insn_bits),
        instruction_formats::OPCODE_STORE => process_opcode_store(processor, insn_bits),
        instruction_formats::OPCODE_JAL => {
            Some(processor.process_jal(instruction_formats::JType::new(insn_bits)))
        }
        instruction_formats::OPCODE_JALR => {
            Some(processor.process_jalr(instruction_formats::IType::new(insn_bits)))
        }
        instruction_formats::OPCODE_MISC_MEM => {
            let dec_insn = instruction_formats::IType::new(insn_bits);
            match dec_insn.funct3 {
                0b000 => Some(processor.process_fence(dec_insn)),
                _ => None,
            }
        }
        instruction_formats::OPCODE_SYSTEM => process_opcode_system(processor, insn_bits),
        _ => None,
    }
}
