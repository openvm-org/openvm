// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

pub mod instruction_formats;
pub mod process_instruction;

pub use process_instruction::process_instruction;

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

    fn process_mul(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulh(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhsu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_div(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_divu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_rem(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_remu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_fence(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_csrrw(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrs(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrc(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrwi(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrsi(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrci(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;

    fn process_ecall(&mut self) -> Self::InstructionResult;
    fn process_ebreak(&mut self) -> Self::InstructionResult;
    fn process_wfi(&mut self) -> Self::InstructionResult;
    fn process_mret(&mut self) -> Self::InstructionResult;
}
