// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{instruction_formats, InstructionProcessor};

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
        0b001 => {
            let dec_insn_shamt = instruction_formats::ITypeShamt::new(insn_bits);
            match dec_insn_shamt.funct6 {
                0b000_000 => Some(processor.process_slli(dec_insn_shamt)),
                _ => None,
            }
        }
        0b010 => Some(processor.process_slti(dec_insn)),
        0b011 => Some(processor.process_sltui(dec_insn)),
        0b100 => Some(processor.process_xori(dec_insn)),
        0b101 => {
            let dec_insn_shamt = instruction_formats::ITypeShamt::new(insn_bits);
            match dec_insn_shamt.funct6 {
                0b000_000 => Some(processor.process_srli(dec_insn_shamt)),
                0b010_000 => Some(processor.process_srai(dec_insn_shamt)),
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
        0b011 => Some(processor.process_ld(dec_insn)),
        0b100 => Some(processor.process_lbu(dec_insn)),
        0b101 => Some(processor.process_lhu(dec_insn)),
        0b110 => Some(processor.process_lwu(dec_insn)),
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
        0b011 => Some(processor.process_sd(dec_insn)),
        _ => None,
    }
}

fn process_opcode_op_imm_32<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::IType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_addiw(dec_insn)),
        0b001 => {
            let dec_insn_shamt = instruction_formats::ITypeShamt::new(insn_bits);
            if dec_insn_shamt.funct6 == 0b000_000 && dec_insn_shamt.shamt < 32 {
                Some(processor.process_slliw(dec_insn_shamt))
            } else {
                None
            }
        }
        0b101 => {
            let dec_insn_shamt = instruction_formats::ITypeShamt::new(insn_bits);
            if dec_insn_shamt.shamt >= 32 {
                return None;
            }
            match dec_insn_shamt.funct6 {
                0b000_000 => Some(processor.process_srliw(dec_insn_shamt)),
                0b010_000 => Some(processor.process_sraiw(dec_insn_shamt)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn process_opcode_op_32<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_formats::RType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_addw(dec_insn)),
            0b000_0001 => Some(processor.process_mulw(dec_insn)),
            0b010_0000 => Some(processor.process_subw(dec_insn)),
            _ => None,
        },
        0b001 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sllw(dec_insn)),
            _ => None,
        },
        0b100 => match dec_insn.funct7 {
            0b000_0001 => Some(processor.process_divw(dec_insn)),
            _ => None,
        },
        0b101 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_srlw(dec_insn)),
            0b000_0001 => Some(processor.process_divuw(dec_insn)),
            0b010_0000 => Some(processor.process_sraw(dec_insn)),
            _ => None,
        },
        0b110 => match dec_insn.funct7 {
            0b000_0001 => Some(processor.process_remw(dec_insn)),
            _ => None,
        },
        0b111 => match dec_insn.funct7 {
            0b000_0001 => Some(processor.process_remuw(dec_insn)),
            _ => None,
        },
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
            let dec_insn = instruction_formats::IType::new(insn_bits);
            match dec_insn.funct3 {
                0b000 => Some(processor.process_jalr(dec_insn)),
                _ => None,
            }
        }
        instruction_formats::OPCODE_MISC_MEM => {
            let dec_insn = instruction_formats::IType::new(insn_bits);
            match dec_insn.funct3 {
                0b000 => Some(processor.process_fence(dec_insn)),
                _ => None,
            }
        }
        instruction_formats::OPCODE_OP_IMM_32 => process_opcode_op_imm_32(processor, insn_bits),
        instruction_formats::OPCODE_OP_32 => process_opcode_op_32(processor, insn_bits),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_formats::*, test_helpers::*};

    // A recording `InstructionProcessor` for testing the dispatch tree.
    //
    // Each trait method is implemented as a one-liner that wraps its decoded
    // argument in the matching `Called` variant. Because the trait's
    // `InstructionResult` associated type is `Called`, the return value of
    // `process_instruction(&mut Recorder, bits)` directly answers both:
    //   - did dispatch happen? (`Some` vs `None`), and
    //   - to which method, with what fields? (the variant).
    //
    // The `recorder!` macro below makes the (variant, method, format-type)
    // mapping a single-line table so it can be audited against `lib.rs` at a
    // glance and so a misaligned row is impossible.
    macro_rules! recorder {
        ($($variant:ident $method:ident $ty:ty),* $(,)?) => {
            #[derive(Debug, PartialEq)]
            enum Called {
                $($variant($ty)),*
            }

            struct Recorder;

            impl InstructionProcessor for Recorder {
                type InstructionResult = Called;
                $(
                    fn $method(&mut self, dec_insn: $ty) -> Called {
                        Called::$variant(dec_insn)
                    }
                )*
            }
        };
    }

    recorder! {
        // RV64I base, register-register (R-type, OPCODE_OP = 0x33)
        Add     process_add     RType,
        Sub     process_sub     RType,
        Sll     process_sll     RType,
        Slt     process_slt     RType,
        Sltu    process_sltu    RType,
        Xor     process_xor     RType,
        Srl     process_srl     RType,
        Sra     process_sra     RType,
        Or      process_or      RType,
        And     process_and     RType,

        // RV64I base, register-immediate (I/ITypeShamt, OPCODE_OP_IMM = 0x13)
        Addi    process_addi    IType,
        Slli    process_slli    ITypeShamt,
        Slti    process_slti    IType,
        Sltui   process_sltui   IType,
        Xori    process_xori    IType,
        Srli    process_srli    ITypeShamt,
        Srai    process_srai    ITypeShamt,
        Ori     process_ori     IType,
        Andi    process_andi    IType,

        // U-type
        Lui     process_lui     UType,
        Auipc   process_auipc   UType,

        // Branches (B-type, OPCODE_BRANCH = 0x63)
        Beq     process_beq     BType,
        Bne     process_bne     BType,
        Blt     process_blt     BType,
        Bltu    process_bltu    BType,
        Bge     process_bge     BType,
        Bgeu    process_bgeu    BType,

        // Loads (I-type, OPCODE_LOAD = 0x03)
        Lb      process_lb      IType,
        Lbu     process_lbu     IType,
        Lh      process_lh      IType,
        Lhu     process_lhu     IType,
        Lw      process_lw      IType,

        // Stores (S-type, OPCODE_STORE = 0x23)
        Sb      process_sb      SType,
        Sh      process_sh      SType,
        Sw      process_sw      SType,

        // Jumps
        Jal     process_jal     JType,
        Jalr    process_jalr    IType,

        // Memory ordering
        Fence   process_fence   IType,

        // M extension on OP (0x33)
        Mul     process_mul     RType,
        Mulh    process_mulh    RType,
        Mulhu   process_mulhu   RType,
        Mulhsu  process_mulhsu  RType,
        Div     process_div     RType,
        Divu    process_divu    RType,
        Rem     process_rem     RType,
        Remu    process_remu    RType,

        // RV64I-only loads/stores
        Lwu     process_lwu     IType,
        Ld      process_ld      IType,
        Sd      process_sd      SType,

        // RV64I W-form register-register (OPCODE_OP_32 = 0x3B)
        Addw    process_addw    RType,
        Subw    process_subw    RType,
        Sllw    process_sllw    RType,
        Srlw    process_srlw    RType,
        Sraw    process_sraw    RType,

        // RV64I W-form register-immediate (OPCODE_OP_IMM_32 = 0x1B)
        Addiw   process_addiw   IType,
        Slliw   process_slliw   ITypeShamt,
        Srliw   process_srliw   ITypeShamt,
        Sraiw   process_sraiw   ITypeShamt,

        // RV64M W-form
        Mulw    process_mulw    RType,
        Divw    process_divw    RType,
        Divuw   process_divuw   RType,
        Remw    process_remw    RType,
        Remuw   process_remuw   RType,
    }

    // Per-format dispatch helpers. Each `check_*_dispatch` builds an
    // instruction word, runs it through `process_instruction`, and asserts
    // that the right `Called::*` variant is produced with the right
    // decoded fields. Tests below call these with concrete values and the
    // expected variant constructor.

    fn check_r_dispatch(
        opcode: u32,
        funct7: u32,
        funct3: u32,
        rd: u32,
        rs1: u32,
        rs2: u32,
        expected: fn(RType) -> Called,
    ) {
        let bits = enc_r(opcode, funct7, rs2, rs1, funct3, rd);
        let dec = RType {
            funct7,
            rs2: rs2 as usize,
            rs1: rs1 as usize,
            funct3,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_i_dispatch(
        opcode: u32,
        funct3: u32,
        rd: u32,
        rs1: u32,
        imm: i32,
        expected: fn(IType) -> Called,
    ) {
        let bits = enc_i(opcode, imm, rs1, funct3, rd);
        let dec = IType {
            imm,
            rs1: rs1 as usize,
            funct3,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_i_shamt6_dispatch(
        opcode: u32,
        funct6: u32,
        funct3: u32,
        rd: u32,
        rs1: u32,
        shamt: u32,
        expected: fn(ITypeShamt) -> Called,
    ) {
        let bits = enc_i_shamt6(opcode, funct6, shamt, rs1, funct3, rd);
        let dec = ITypeShamt {
            funct6,
            shamt,
            rs1: rs1 as usize,
            funct3,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    /// W-form shift dispatch. Caller supplies `funct7` (the field the spec
    /// uses for W shifts), but the decoder produces `funct6 = funct7 >> 1`
    /// because it reads bits 31:26 unconditionally. Caller must keep
    /// `shamt < 32`.
    fn check_i_shamt5_dispatch(
        opcode: u32,
        funct7: u32,
        funct3: u32,
        rd: u32,
        rs1: u32,
        shamt: u32,
        expected: fn(ITypeShamt) -> Called,
    ) {
        let bits = enc_i_shamt5(opcode, funct7, shamt, rs1, funct3, rd);
        let dec = ITypeShamt {
            funct6: funct7 >> 1,
            shamt,
            rs1: rs1 as usize,
            funct3,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_s_dispatch(
        opcode: u32,
        funct3: u32,
        rs1: u32,
        rs2: u32,
        imm: i32,
        expected: fn(SType) -> Called,
    ) {
        let bits = enc_s(opcode, imm, rs2, rs1, funct3);
        let dec = SType {
            imm,
            rs2: rs2 as usize,
            rs1: rs1 as usize,
            funct3,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_b_dispatch(
        opcode: u32,
        funct3: u32,
        rs1: u32,
        rs2: u32,
        imm: i32,
        expected: fn(BType) -> Called,
    ) {
        let bits = enc_b(opcode, imm, rs2, rs1, funct3);
        let dec = BType {
            imm,
            rs2: rs2 as usize,
            rs1: rs1 as usize,
            funct3,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_u_dispatch(opcode: u32, rd: u32, imm: i32, expected: fn(UType) -> Called) {
        let bits = enc_u(opcode, imm, rd);
        let dec = UType {
            imm,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    fn check_j_dispatch(opcode: u32, rd: u32, imm: i32, expected: fn(JType) -> Called) {
        let bits = enc_j(opcode, imm, rd);
        let dec = JType {
            imm,
            rd: rd as usize,
        };
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(expected(dec))
        );
    }

    /// Asserts that `process_instruction` rejects the given encoding (returns `None`).
    fn check_rejects(bits: u32) {
        assert_eq!(process_instruction(&mut Recorder, bits), None);
    }

    // ---- Happy-path dispatch tests, one function per opcode group ------

    #[test]
    fn dispatch_op_r_type() {
        // Base RV64I R-type integer ops on OPCODE_OP = 0x33.
        check_r_dispatch(OPCODE_OP, 0, 0, 1, 2, 3, Called::Add);
        check_r_dispatch(OPCODE_OP, 0x20, 0, 4, 5, 6, Called::Sub);
        check_r_dispatch(OPCODE_OP, 0, 1, 7, 8, 9, Called::Sll);
        check_r_dispatch(OPCODE_OP, 0, 2, 10, 11, 12, Called::Slt);
        check_r_dispatch(OPCODE_OP, 0, 3, 13, 14, 15, Called::Sltu);
        check_r_dispatch(OPCODE_OP, 0, 4, 16, 17, 18, Called::Xor);
        check_r_dispatch(OPCODE_OP, 0, 5, 19, 20, 21, Called::Srl);
        check_r_dispatch(OPCODE_OP, 0x20, 5, 22, 23, 24, Called::Sra);
        check_r_dispatch(OPCODE_OP, 0, 6, 25, 26, 27, Called::Or);
        check_r_dispatch(OPCODE_OP, 0, 7, 28, 29, 30, Called::And);
    }

    #[test]
    fn dispatch_op_m_extension() {
        // M-extension R-type ops on OPCODE_OP (funct7 = 1).
        check_r_dispatch(OPCODE_OP, 1, 0, 31, 0, 1, Called::Mul);
        check_r_dispatch(OPCODE_OP, 1, 1, 2, 3, 4, Called::Mulh);
        check_r_dispatch(OPCODE_OP, 1, 2, 5, 6, 7, Called::Mulhsu);
        check_r_dispatch(OPCODE_OP, 1, 3, 8, 9, 10, Called::Mulhu);
        check_r_dispatch(OPCODE_OP, 1, 4, 11, 12, 13, Called::Div);
        check_r_dispatch(OPCODE_OP, 1, 5, 14, 15, 16, Called::Divu);
        check_r_dispatch(OPCODE_OP, 1, 6, 17, 18, 19, Called::Rem);
        check_r_dispatch(OPCODE_OP, 1, 7, 20, 21, 22, Called::Remu);
    }

    #[test]
    fn dispatch_op_imm() {
        // I-type ops on OPCODE_OP_IMM = 0x13.
        check_i_dispatch(OPCODE_OP_IMM, 0, 23, 24, 42, Called::Addi);
        check_i_dispatch(OPCODE_OP_IMM, 2, 25, 26, -5, Called::Slti);
        check_i_dispatch(OPCODE_OP_IMM, 3, 27, 28, 100, Called::Sltui);
        check_i_dispatch(OPCODE_OP_IMM, 4, 29, 30, -100, Called::Xori);
        check_i_dispatch(OPCODE_OP_IMM, 6, 31, 0, 2047, Called::Ori);
        check_i_dispatch(OPCODE_OP_IMM, 7, 1, 2, -2048, Called::Andi);
        // Shift-immediate forms use RV64 6-bit shamt + 6-bit funct6.
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0, 1, 3, 4, 5, Called::Slli);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0, 5, 6, 7, 33, Called::Srli);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x10, 5, 8, 9, 63, Called::Srai);
    }

    #[test]
    fn dispatch_u_type() {
        check_u_dispatch(OPCODE_LUI, 10, 0x12345000_u32 as i32, Called::Lui);
        // imm with high bit set exercises the sign-extended case.
        check_u_dispatch(OPCODE_AUIPC, 11, 0xabcde000_u32 as i32, Called::Auipc);
    }

    #[test]
    fn dispatch_branches() {
        check_b_dispatch(OPCODE_BRANCH, 0, 12, 13, 100, Called::Beq);
        check_b_dispatch(OPCODE_BRANCH, 1, 14, 15, -200, Called::Bne);
        check_b_dispatch(OPCODE_BRANCH, 4, 16, 17, 4094, Called::Blt);
        check_b_dispatch(OPCODE_BRANCH, 5, 18, 19, -4096, Called::Bge);
        check_b_dispatch(OPCODE_BRANCH, 6, 20, 21, 2, Called::Bltu);
        check_b_dispatch(OPCODE_BRANCH, 7, 22, 23, -2, Called::Bgeu);
    }

    #[test]
    fn dispatch_loads() {
        check_i_dispatch(OPCODE_LOAD, 0, 24, 25, 10, Called::Lb);
        check_i_dispatch(OPCODE_LOAD, 1, 26, 27, 20, Called::Lh);
        check_i_dispatch(OPCODE_LOAD, 2, 28, 29, -50, Called::Lw);
        check_i_dispatch(OPCODE_LOAD, 3, 30, 31, 1000, Called::Ld);
        check_i_dispatch(OPCODE_LOAD, 4, 0, 1, -1, Called::Lbu);
        check_i_dispatch(OPCODE_LOAD, 5, 2, 3, 0, Called::Lhu);
        check_i_dispatch(OPCODE_LOAD, 6, 4, 5, 2047, Called::Lwu);
    }

    #[test]
    fn dispatch_stores() {
        check_s_dispatch(OPCODE_STORE, 0, 6, 7, 8, Called::Sb);
        check_s_dispatch(OPCODE_STORE, 1, 8, 9, -8, Called::Sh);
        check_s_dispatch(OPCODE_STORE, 2, 10, 11, 2047, Called::Sw);
        check_s_dispatch(OPCODE_STORE, 3, 12, 13, -2048, Called::Sd);
    }

    #[test]
    fn dispatch_jumps() {
        check_j_dispatch(OPCODE_JAL, 14, 2046, Called::Jal);
        // JALR uses I-type encoding with funct3=0 (spec §2.5.1).
        check_i_dispatch(OPCODE_JALR, 0, 15, 16, 42, Called::Jalr);
    }

    #[test]
    fn dispatch_fence() {
        // FENCE encodes fm/pred/succ in the imm field; the decoder treats
        // the field as an I-type immediate without validating its contents.
        check_i_dispatch(OPCODE_MISC_MEM, 0, 17, 18, 0x0ff, Called::Fence);
    }

    #[test]
    fn dispatch_op_32_r_type() {
        // RV64 W-form base ops on OPCODE_OP_32 = 0x3B.
        check_r_dispatch(OPCODE_OP_32, 0, 0, 19, 20, 21, Called::Addw);
        check_r_dispatch(OPCODE_OP_32, 0x20, 0, 22, 23, 24, Called::Subw);
        check_r_dispatch(OPCODE_OP_32, 0, 1, 25, 26, 27, Called::Sllw);
        check_r_dispatch(OPCODE_OP_32, 0, 5, 28, 29, 30, Called::Srlw);
        check_r_dispatch(OPCODE_OP_32, 0x20, 5, 31, 0, 1, Called::Sraw);
    }

    #[test]
    fn dispatch_op_32_m_extension() {
        // RV64M W-form ops (funct7 = 1).
        check_r_dispatch(OPCODE_OP_32, 1, 0, 2, 3, 4, Called::Mulw);
        check_r_dispatch(OPCODE_OP_32, 1, 4, 5, 6, 7, Called::Divw);
        check_r_dispatch(OPCODE_OP_32, 1, 5, 8, 9, 10, Called::Divuw);
        check_r_dispatch(OPCODE_OP_32, 1, 6, 11, 12, 13, Called::Remw);
        check_r_dispatch(OPCODE_OP_32, 1, 7, 14, 15, 16, Called::Remuw);
    }

    #[test]
    fn dispatch_op_imm_32() {
        check_i_dispatch(OPCODE_OP_IMM_32, 0, 17, 18, 100, Called::Addiw);
        // W-form shifts use 5-bit shamt; shamt < 32 is required.
        check_i_shamt5_dispatch(OPCODE_OP_IMM_32, 0, 1, 19, 20, 5, Called::Slliw);
        check_i_shamt5_dispatch(OPCODE_OP_IMM_32, 0, 5, 21, 22, 31, Called::Srliw);
        check_i_shamt5_dispatch(OPCODE_OP_IMM_32, 0x20, 5, 23, 24, 0, Called::Sraiw);
    }

    // ---- Rejection-path tests, one function per opcode group ----------

    #[test]
    fn rejects_op_invalid_funct7() {
        // OPCODE_OP accepts funct7 in {0, 1, 0x20}; everything else is illegal.
        // funct7 = 2 is outside that set, regardless of funct3.
        for funct3 in 0..8 {
            check_rejects(enc_r(OPCODE_OP, 0x02, 1, 2, funct3, 3));
        }
    }

    #[test]
    fn rejects_op_imm_shifts_invalid_funct6() {
        // SLLI requires funct6 = 0; SRLI/SRAI require funct6 in {0, 0x10}.
        check_rejects(enc_i_shamt6(OPCODE_OP_IMM, 0x01, 5, 1, 1, 2));
        check_rejects(enc_i_shamt6(OPCODE_OP_IMM, 0x01, 5, 1, 5, 2));
    }

    #[test]
    fn rejects_branch_unassigned_funct3() {
        // BRANCH funct3 in {2, 3} are unassigned.
        check_rejects(enc_b(OPCODE_BRANCH, 0, 1, 2, 2));
        check_rejects(enc_b(OPCODE_BRANCH, 0, 1, 2, 3));
    }

    #[test]
    fn rejects_load_unassigned_funct3() {
        // LD uses funct3=3, LWU uses funct3=6; funct3=7 is unassigned.
        check_rejects(enc_i(OPCODE_LOAD, 0, 1, 7, 2));
    }

    #[test]
    fn rejects_store_unassigned_funct3() {
        // STORE funct3 in {4, 5, 6, 7} are unassigned.
        for funct3 in 4..8 {
            check_rejects(enc_s(OPCODE_STORE, 0, 1, 2, funct3));
        }
    }

    #[test]
    fn rejects_jalr_with_nonzero_funct3() {
        // Per spec §2.5.1, JALR requires funct3=0.
        check_rejects(enc_i(OPCODE_JALR, 0, 5, 1, 4));
    }

    #[test]
    fn rejects_fence_i() {
        // FENCE.I (Zifencei) has opcode 0x0F, funct3=1. This decoder does
        // not support Zifencei and rejects it.
        check_rejects(enc_i(OPCODE_MISC_MEM, 0, 0, 1, 0));
    }

    #[test]
    fn rejects_op_32_invalid_funct7() {
        // OPCODE_OP_32 funct3 in {0, 1, 4, 5, 6, 7} each have specific
        // funct7 values defined; funct7=2 is outside the valid set for all.
        for funct3 in [0, 1, 4, 5, 6, 7] {
            check_rejects(enc_r(OPCODE_OP_32, 0x02, 1, 2, funct3, 3));
        }
    }

    #[test]
    fn rejects_op_32_unassigned_funct3() {
        // OPCODE_OP_32 funct3 in {2, 3} have no defined W-form instructions
        // (no SLTW / SLTUW etc.).
        check_rejects(enc_r(OPCODE_OP_32, 0, 1, 2, 2, 3));
        check_rejects(enc_r(OPCODE_OP_32, 0, 1, 2, 3, 3));
    }

    #[test]
    fn rejects_op_imm_32_invalid_shifts() {
        // W-form shifts use 5-bit shamt; shamt >= 32 is illegal.
        check_rejects(enc_i_shamt5(OPCODE_OP_IMM_32, 0, 32, 1, 1, 2));
        check_rejects(enc_i_shamt5(OPCODE_OP_IMM_32, 0, 32, 1, 5, 2));
        // SLLIW must have bits 31:25 = 0; funct7=2 sets bit 26 (funct6=1).
        check_rejects(enc_i_shamt5(OPCODE_OP_IMM_32, 0x02, 5, 1, 1, 2));
        // SRLIW/SRAIW must have funct6 in {0, 0x10}.
        check_rejects(enc_i_shamt5(OPCODE_OP_IMM_32, 0x02, 5, 1, 5, 2));
    }

    #[test]
    fn rejects_op_imm_32_unassigned_funct3() {
        // OPCODE_OP_IMM_32 defines only ADDIW (funct3=0), SLLIW (1),
        // SRLIW/SRAIW (5). All other funct3 values are unassigned.
        for funct3 in [2, 3, 4, 6, 7] {
            check_rejects(enc_i(OPCODE_OP_IMM_32, 0, 1, funct3, 2));
        }
    }

    #[test]
    fn rejects_system_opcode() {
        // SYSTEM opcode 0x73 (ECALL/EBREAK/CSRs) is not in this decoder's
        // dispatch tree. The transpiler layer handles these earlier.
        check_rejects(0x00100073); // ebreak
        check_rejects(0x30001073); // csrrw x0, mstatus, x0
    }
}
