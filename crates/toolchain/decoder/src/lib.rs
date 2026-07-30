// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

pub mod instruction_formats;
pub mod process_instruction;

#[cfg(test)]
mod test_helpers;

pub use process_instruction::{is_bitmanip_instruction, process_instruction};

/// A trait for objects which do something with RISC-V instructions (e.g. execute them or print a
/// disassembly string).
///
/// There is one function per RISC-V instruction. Each function takes the appropriate struct from
/// [instruction_formats] giving access to the decoded fields of the instruction. All functions
/// return the [InstructionProcessor::InstructionResult] associated type.
pub trait InstructionProcessor {
    type InstructionResult;

    fn process_add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sub(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sll(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_slt(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sltu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_xor(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_srl(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sra(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_or(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_and(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_addi(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_slli(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_slti(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_sltui(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_xori(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_srli(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_srai(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_ori(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_andi(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_lui(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;
    fn process_auipc(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;

    fn process_beq(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bne(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_blt(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bltu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bge(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bgeu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;

    fn process_lb(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lbu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lh(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lhu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_sb(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;
    fn process_sh(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;
    fn process_sw(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;

    fn process_jal(&mut self, dec_insn: instruction_formats::JType) -> Self::InstructionResult;
    fn process_jalr(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_fence(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_mul(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulh(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhsu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_div(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_divu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_rem(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_remu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_lwu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_ld(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_sd(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;

    fn process_addw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_subw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sllw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_srlw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sraw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_addiw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_slliw(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_srliw(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_sraiw(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;

    fn process_mulw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_divw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_divuw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_remw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_remuw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // Zba (address generation)
    fn process_add_uw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sh1add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sh2add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sh3add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sh1add_uw(
        &mut self,
        dec_insn: instruction_formats::RType,
    ) -> Self::InstructionResult;
    fn process_sh2add_uw(
        &mut self,
        dec_insn: instruction_formats::RType,
    ) -> Self::InstructionResult;
    fn process_sh3add_uw(
        &mut self,
        dec_insn: instruction_formats::RType,
    ) -> Self::InstructionResult;
    fn process_slli_uw(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;

    // Zbb: logical with negate
    fn process_andn(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_orn(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_xnor(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // Zbb: counts. These are unary (rd, rs1); the sub-operation is encoded in the
    // rs2/shamt field of an I-type word, so the `imm` field carries the full funct12.
    fn process_clz(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_ctz(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_cpop(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_clzw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_ctzw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_cpopw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    // Zbb: min/max
    fn process_min(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_minu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_max(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_maxu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // Zbb: sign/zero extension (unary; zext.h is R-type with rs2 = 0)
    fn process_sext_b(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_sext_h(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_zext_h(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // Zbb: rotates
    fn process_rol(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_ror(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_rori(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_rolw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_rorw(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_roriw(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;

    // Zbb: byte ops (unary)
    fn process_orc_b(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_rev8(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    // Zbs (single-bit)
    fn process_bclr(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_bset(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_binv(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_bext(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_bclri(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_bseti(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_binvi(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_bexti(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
}
