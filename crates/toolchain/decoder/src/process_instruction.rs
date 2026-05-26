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
            Some(processor.process_jalr(instruction_formats::IType::new(insn_bits)))
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
    use crate::instruction_formats::*;

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
        // RV32I + RV64I base, register-register (R-type, OPCODE_OP = 0x33)
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

        // RV32I + RV64I base, register-immediate (I/ITypeShamt, OPCODE_OP_IMM = 0x13)
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

    // Encoder helpers built directly from the RISC-V Unprivileged Spec bit
    // layouts. Each `enc_*` function constructs the canonical bit pattern for
    // its format; the per-format `test_*!` macros below combine an encoder
    // call with a `process_instruction` assertion so each test reads as a
    // single tabular row.

    fn enc_r(opcode: u32, funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn enc_i(opcode: u32, imm: i32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        let imm_u = (imm as u32) & 0xfff;
        (imm_u << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    // RV64 OP_IMM shifts: funct6 in bits 31:26, shamt 6 bits in bits 25:20.
    fn enc_i_shamt6(opcode: u32, funct6: u32, shamt: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct6 << 26) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    // RV64 OP_IMM_32 (W-form) shifts: funct7 in bits 31:25, shamt 5 bits in bits 24:20.
    // Bit 25 must be 0 (encoded via funct7 having a 0 low bit).
    fn enc_i_shamt5(opcode: u32, funct7: u32, shamt: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct7 << 25) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn enc_s(opcode: u32, imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm_u = (imm as u32) & 0xfff;
        let imm_hi = (imm_u >> 5) & 0x7f;
        let imm_lo = imm_u & 0x1f;
        (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode
    }

    fn enc_b(opcode: u32, imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm_u = (imm as u32) & 0x1fff;
        let bit12 = (imm_u >> 12) & 1;
        let bit11 = (imm_u >> 11) & 1;
        let bits_10_5 = (imm_u >> 5) & 0x3f;
        let bits_4_1 = (imm_u >> 1) & 0xf;
        (bit12 << 31)
            | (bits_10_5 << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (bits_4_1 << 8)
            | (bit11 << 7)
            | opcode
    }

    fn enc_u(opcode: u32, imm: i32, rd: u32) -> u32 {
        ((imm as u32) & 0xffff_f000) | (rd << 7) | opcode
    }

    fn enc_j(opcode: u32, imm: i32, rd: u32) -> u32 {
        let imm_u = (imm as u32) & 0x1f_ffff;
        let bit20 = (imm_u >> 20) & 1;
        let bits_10_1 = (imm_u >> 1) & 0x3ff;
        let bit11 = (imm_u >> 11) & 1;
        let bits_19_12 = (imm_u >> 12) & 0xff;
        (bit20 << 31)
            | (bits_10_1 << 21)
            | (bit11 << 20)
            | (bits_19_12 << 12)
            | (rd << 7)
            | opcode
    }

    // Per-format test macros. Each row passes its own register / imm values
    // so the suite as a whole varies field values across tests.

    macro_rules! test_r {
        ($name:ident => $variant:ident, opcode = $op:expr, funct7 = $f7:expr, funct3 = $f3:expr,
         rd = $rd:expr, rs1 = $rs1:expr, rs2 = $rs2:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_r($op, $f7, $rs2 as u32, $rs1 as u32, $f3, $rd as u32);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(RType {
                        funct7: $f7,
                        rs2: $rs2 as usize,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                        rd: $rd as usize,
                    }))
                );
            }
        };
    }

    macro_rules! test_i {
        ($name:ident => $variant:ident, opcode = $op:expr, funct3 = $f3:expr,
         rd = $rd:expr, rs1 = $rs1:expr, imm = $imm:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_i($op, $imm, $rs1 as u32, $f3, $rd as u32);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(IType {
                        imm: $imm,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                        rd: $rd as usize,
                    }))
                );
            }
        };
    }

    macro_rules! test_i_shamt6 {
        ($name:ident => $variant:ident, opcode = $op:expr, funct6 = $f6:expr, funct3 = $f3:expr,
         rd = $rd:expr, rs1 = $rs1:expr, shamt = $shamt:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_i_shamt6($op, $f6, $shamt, $rs1 as u32, $f3, $rd as u32);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(ITypeShamt {
                        funct6: $f6,
                        shamt: $shamt,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                        rd: $rd as usize,
                    }))
                );
            }
        };
    }

    // For OP_IMM_32 shifts, the decoder still produces an `ITypeShamt`; its
    // `funct6` field is bits 31:26. With shamt < 32 (bit 25 = 0) and funct7
    // having a low bit of 0, the produced funct6 is `funct7 >> 1`.
    macro_rules! test_i_shamt5 {
        ($name:ident => $variant:ident, opcode = $op:expr, funct7 = $f7:expr, funct3 = $f3:expr,
         rd = $rd:expr, rs1 = $rs1:expr, shamt = $shamt:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_i_shamt5($op, $f7, $shamt, $rs1 as u32, $f3, $rd as u32);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(ITypeShamt {
                        funct6: $f7 >> 1,
                        shamt: $shamt,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                        rd: $rd as usize,
                    }))
                );
            }
        };
    }

    macro_rules! test_s {
        ($name:ident => $variant:ident, opcode = $op:expr, funct3 = $f3:expr,
         rs1 = $rs1:expr, rs2 = $rs2:expr, imm = $imm:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_s($op, $imm, $rs2 as u32, $rs1 as u32, $f3);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(SType {
                        imm: $imm,
                        rs2: $rs2 as usize,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                    }))
                );
            }
        };
    }

    macro_rules! test_b {
        ($name:ident => $variant:ident, opcode = $op:expr, funct3 = $f3:expr,
         rs1 = $rs1:expr, rs2 = $rs2:expr, imm = $imm:expr $(,)?) => {
            #[test]
            fn $name() {
                let bits = enc_b($op, $imm, $rs2 as u32, $rs1 as u32, $f3);
                assert_eq!(
                    process_instruction(&mut Recorder, bits),
                    Some(Called::$variant(BType {
                        imm: $imm,
                        rs2: $rs2 as usize,
                        rs1: $rs1 as usize,
                        funct3: $f3,
                    }))
                );
            }
        };
    }

    macro_rules! test_rejects {
        ($name:ident, $bits:expr $(,)?) => {
            #[test]
            fn $name() {
                assert_eq!(process_instruction(&mut Recorder, $bits), None);
            }
        };
    }

    // ---- OP (0x33) -- R-type -------------------------------------------
    test_r!(dispatch_add    => Add,    opcode = 0x33, funct7 = 0,    funct3 = 0, rd = 1,  rs1 = 2,  rs2 = 3);
    test_r!(dispatch_sub    => Sub,    opcode = 0x33, funct7 = 0x20, funct3 = 0, rd = 4,  rs1 = 5,  rs2 = 6);
    test_r!(dispatch_sll    => Sll,    opcode = 0x33, funct7 = 0,    funct3 = 1, rd = 7,  rs1 = 8,  rs2 = 9);
    test_r!(dispatch_slt    => Slt,    opcode = 0x33, funct7 = 0,    funct3 = 2, rd = 10, rs1 = 11, rs2 = 12);
    test_r!(dispatch_sltu   => Sltu,   opcode = 0x33, funct7 = 0,    funct3 = 3, rd = 13, rs1 = 14, rs2 = 15);
    test_r!(dispatch_xor    => Xor,    opcode = 0x33, funct7 = 0,    funct3 = 4, rd = 16, rs1 = 17, rs2 = 18);
    test_r!(dispatch_srl    => Srl,    opcode = 0x33, funct7 = 0,    funct3 = 5, rd = 19, rs1 = 20, rs2 = 21);
    test_r!(dispatch_sra    => Sra,    opcode = 0x33, funct7 = 0x20, funct3 = 5, rd = 22, rs1 = 23, rs2 = 24);
    test_r!(dispatch_or     => Or,     opcode = 0x33, funct7 = 0,    funct3 = 6, rd = 25, rs1 = 26, rs2 = 27);
    test_r!(dispatch_and    => And,    opcode = 0x33, funct7 = 0,    funct3 = 7, rd = 28, rs1 = 29, rs2 = 30);
    test_rejects!(rejects_op_funct3_0_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 0, 3));
    test_rejects!(rejects_op_funct3_1_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 1, 3));
    test_rejects!(rejects_op_funct3_2_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 2, 3));
    test_rejects!(rejects_op_funct3_3_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 3, 3));
    test_rejects!(rejects_op_funct3_4_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 4, 3));
    test_rejects!(rejects_op_funct3_5_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 5, 3));
    test_rejects!(rejects_op_funct3_6_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 6, 3));
    test_rejects!(rejects_op_funct3_7_invalid_funct7, enc_r(OPCODE_OP, 0x02, 1, 2, 7, 3));

    // ---- OP (0x33) -- M extension --------------------------------------
    test_r!(dispatch_mul    => Mul,    opcode = 0x33, funct7 = 1, funct3 = 0, rd = 31, rs1 = 0,  rs2 = 1);
    test_r!(dispatch_mulh   => Mulh,   opcode = 0x33, funct7 = 1, funct3 = 1, rd = 2,  rs1 = 3,  rs2 = 4);
    test_r!(dispatch_mulhsu => Mulhsu, opcode = 0x33, funct7 = 1, funct3 = 2, rd = 5,  rs1 = 6,  rs2 = 7);
    test_r!(dispatch_mulhu  => Mulhu,  opcode = 0x33, funct7 = 1, funct3 = 3, rd = 8,  rs1 = 9,  rs2 = 10);
    test_r!(dispatch_div    => Div,    opcode = 0x33, funct7 = 1, funct3 = 4, rd = 11, rs1 = 12, rs2 = 13);
    test_r!(dispatch_divu   => Divu,   opcode = 0x33, funct7 = 1, funct3 = 5, rd = 14, rs1 = 15, rs2 = 16);
    test_r!(dispatch_rem    => Rem,    opcode = 0x33, funct7 = 1, funct3 = 6, rd = 17, rs1 = 18, rs2 = 19);
    test_r!(dispatch_remu   => Remu,   opcode = 0x33, funct7 = 1, funct3 = 7, rd = 20, rs1 = 21, rs2 = 22);

    // ---- OP_IMM (0x13) -- I-type / ITypeShamt --------------------------
    test_i!(dispatch_addi  => Addi,  opcode = 0x13, funct3 = 0, rd = 23, rs1 = 24, imm = 42);
    test_i!(dispatch_slti  => Slti,  opcode = 0x13, funct3 = 2, rd = 25, rs1 = 26, imm = -5);
    test_i!(dispatch_sltui => Sltui, opcode = 0x13, funct3 = 3, rd = 27, rs1 = 28, imm = 100);
    test_i!(dispatch_xori  => Xori,  opcode = 0x13, funct3 = 4, rd = 29, rs1 = 30, imm = -100);
    test_i!(dispatch_ori   => Ori,   opcode = 0x13, funct3 = 6, rd = 31, rs1 = 0,  imm = 2047);
    test_i!(dispatch_andi  => Andi,  opcode = 0x13, funct3 = 7, rd = 1,  rs1 = 2,  imm = -2048);
    test_i_shamt6!(dispatch_slli => Slli, opcode = 0x13, funct6 = 0,    funct3 = 1, rd = 3, rs1 = 4, shamt = 5);
    test_i_shamt6!(dispatch_srli => Srli, opcode = 0x13, funct6 = 0,    funct3 = 5, rd = 6, rs1 = 7, shamt = 33);
    test_i_shamt6!(dispatch_srai => Srai, opcode = 0x13, funct6 = 0x10, funct3 = 5, rd = 8, rs1 = 9, shamt = 63);
    // SLLI has funct6=0 only; anything else is illegal.
    test_rejects!(rejects_slli_invalid_funct6,
        enc_i_shamt6(OPCODE_OP_IMM, 0x01, 5, 1, 1, 2));
    // SRLI uses funct6=0, SRAI uses funct6=0x10; anything else is illegal.
    test_rejects!(rejects_srli_srai_invalid_funct6,
        enc_i_shamt6(OPCODE_OP_IMM, 0x01, 5, 1, 5, 2));

    // ---- U-type --------------------------------------------------------
    #[test]
    fn dispatch_lui() {
        let bits = enc_u(0x37, 0x12345000u32 as i32, 10);
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(Called::Lui(UType {
                imm: 0x12345000,
                rd: 10,
            }))
        );
    }

    #[test]
    fn dispatch_auipc() {
        // imm with high bit set to also exercise the sign-extended case.
        let bits = enc_u(0x17, 0xabcde000u32 as i32, 11);
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(Called::Auipc(UType {
                imm: 0xabcde000u32 as i32,
                rd: 11,
            }))
        );
    }

    // ---- BRANCH (0x63) -- B-type ---------------------------------------
    test_b!(dispatch_beq  => Beq,  opcode = 0x63, funct3 = 0, rs1 = 12, rs2 = 13, imm = 100);
    test_b!(dispatch_bne  => Bne,  opcode = 0x63, funct3 = 1, rs1 = 14, rs2 = 15, imm = -200);
    test_b!(dispatch_blt  => Blt,  opcode = 0x63, funct3 = 4, rs1 = 16, rs2 = 17, imm = 4094);
    test_b!(dispatch_bge  => Bge,  opcode = 0x63, funct3 = 5, rs1 = 18, rs2 = 19, imm = -4096);
    test_b!(dispatch_bltu => Bltu, opcode = 0x63, funct3 = 6, rs1 = 20, rs2 = 21, imm = 2);
    test_b!(dispatch_bgeu => Bgeu, opcode = 0x63, funct3 = 7, rs1 = 22, rs2 = 23, imm = -2);
    test_rejects!(rejects_branch_funct3_2, enc_b(OPCODE_BRANCH, 0, 1, 2, 2));
    test_rejects!(rejects_branch_funct3_3, enc_b(OPCODE_BRANCH, 0, 1, 2, 3));

    // ---- LOAD (0x03) -- I-type -----------------------------------------
    test_i!(dispatch_lb  => Lb,  opcode = 0x03, funct3 = 0, rd = 24, rs1 = 25, imm = 10);
    test_i!(dispatch_lh  => Lh,  opcode = 0x03, funct3 = 1, rd = 26, rs1 = 27, imm = 20);
    test_i!(dispatch_lw  => Lw,  opcode = 0x03, funct3 = 2, rd = 28, rs1 = 29, imm = -50);
    test_i!(dispatch_ld  => Ld,  opcode = 0x03, funct3 = 3, rd = 30, rs1 = 31, imm = 1000);
    test_i!(dispatch_lbu => Lbu, opcode = 0x03, funct3 = 4, rd = 0,  rs1 = 1,  imm = -1);
    test_i!(dispatch_lhu => Lhu, opcode = 0x03, funct3 = 5, rd = 2,  rs1 = 3,  imm = 0);
    test_i!(dispatch_lwu => Lwu, opcode = 0x03, funct3 = 6, rd = 4,  rs1 = 5,  imm = 2047);
    // LD uses 3, LWU uses 6. funct3=7 is unassigned.
    test_rejects!(rejects_load_funct3_7, enc_i(OPCODE_LOAD, 0, 1, 7, 2));

    // ---- STORE (0x23) -- S-type ----------------------------------------
    test_s!(dispatch_sb => Sb, opcode = 0x23, funct3 = 0, rs1 = 6,  rs2 = 7,  imm = 8);
    test_s!(dispatch_sh => Sh, opcode = 0x23, funct3 = 1, rs1 = 8,  rs2 = 9,  imm = -8);
    test_s!(dispatch_sw => Sw, opcode = 0x23, funct3 = 2, rs1 = 10, rs2 = 11, imm = 2047);
    test_s!(dispatch_sd => Sd, opcode = 0x23, funct3 = 3, rs1 = 12, rs2 = 13, imm = -2048);
    test_rejects!(rejects_store_funct3_4, enc_s(OPCODE_STORE, 0, 1, 2, 4));
    test_rejects!(rejects_store_funct3_5, enc_s(OPCODE_STORE, 0, 1, 2, 5));
    test_rejects!(rejects_store_funct3_6, enc_s(OPCODE_STORE, 0, 1, 2, 6));
    test_rejects!(rejects_store_funct3_7, enc_s(OPCODE_STORE, 0, 1, 2, 7));

    // ---- Jumps ---------------------------------------------------------
    #[test]
    fn dispatch_jal() {
        let bits = enc_j(OPCODE_JAL, 2046, 14);
        assert_eq!(
            process_instruction(&mut Recorder, bits),
            Some(Called::Jal(JType { imm: 2046, rd: 14 }))
        );
    }

    // JALR uses I-type encoding. Per spec §2.5.1 (RISC-V Unprivileged
    // Architecture, v20260120), JALR requires funct3=0; the decoder
    // currently dispatches for any funct3. The rejection test below
    // asserts the spec-correct behavior and fails until the decoder is
    // fixed.
    test_i!(dispatch_jalr => Jalr, opcode = 0x67, funct3 = 0, rd = 15, rs1 = 16, imm = 42);
    test_rejects!(
        rejects_jalr_with_nonzero_funct3,
        enc_i(OPCODE_JALR, 0, 5, 1, 4),
    );

    // ---- MISC-MEM (0x0F) -----------------------------------------------
    // FENCE encodes fm/pred/succ in the imm field; for dispatch we just need
    // funct3=0 and an arbitrary imm value (decoder doesn't validate fm/pred/succ).
    test_i!(dispatch_fence => Fence, opcode = 0x0f, funct3 = 0, rd = 17, rs1 = 18, imm = 0x0ff);
    test_rejects!(rejects_fence_i, enc_i(OPCODE_MISC_MEM, 0, 0, 1, 0));

    // ---- OP_32 (0x3B) -- R-type (RV64 only) ----------------------------
    test_r!(dispatch_addw  => Addw,  opcode = 0x3b, funct7 = 0,    funct3 = 0, rd = 19, rs1 = 20, rs2 = 21);
    test_r!(dispatch_subw  => Subw,  opcode = 0x3b, funct7 = 0x20, funct3 = 0, rd = 22, rs1 = 23, rs2 = 24);
    test_r!(dispatch_sllw  => Sllw,  opcode = 0x3b, funct7 = 0,    funct3 = 1, rd = 25, rs1 = 26, rs2 = 27);
    test_r!(dispatch_srlw  => Srlw,  opcode = 0x3b, funct7 = 0,    funct3 = 5, rd = 28, rs1 = 29, rs2 = 30);
    test_r!(dispatch_sraw  => Sraw,  opcode = 0x3b, funct7 = 0x20, funct3 = 5, rd = 31, rs1 = 0,  rs2 = 1);

    // ---- OP_32 (0x3B) -- M extension -----------------------------------
    test_r!(dispatch_mulw  => Mulw,  opcode = 0x3b, funct7 = 1, funct3 = 0, rd = 2,  rs1 = 3,  rs2 = 4);
    test_r!(dispatch_divw  => Divw,  opcode = 0x3b, funct7 = 1, funct3 = 4, rd = 5,  rs1 = 6,  rs2 = 7);
    test_r!(dispatch_divuw => Divuw, opcode = 0x3b, funct7 = 1, funct3 = 5, rd = 8,  rs1 = 9,  rs2 = 10);
    test_r!(dispatch_remw  => Remw,  opcode = 0x3b, funct7 = 1, funct3 = 6, rd = 11, rs1 = 12, rs2 = 13);
    test_r!(dispatch_remuw => Remuw, opcode = 0x3b, funct7 = 1, funct3 = 7, rd = 14, rs1 = 15, rs2 = 16);
    test_rejects!(rejects_op_32_funct3_0_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 0, 3));
    test_rejects!(rejects_op_32_funct3_1_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 1, 3));
    test_rejects!(rejects_op_32_funct3_4_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 4, 3));
    test_rejects!(rejects_op_32_funct3_5_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 5, 3));
    test_rejects!(rejects_op_32_funct3_6_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 6, 3));
    test_rejects!(rejects_op_32_funct3_7_invalid_funct7, enc_r(OPCODE_OP_32, 0x02, 1, 2, 7, 3));
    // OP_32 funct3 in {2, 3} have no defined W-form instructions (no
    // SLTW / SLTUW etc., because comparisons return a single bit that
    // doesn't need a separate W-form).
    test_rejects!(rejects_op_32_funct3_2, enc_r(OPCODE_OP_32, 0, 1, 2, 2, 3));
    test_rejects!(rejects_op_32_funct3_3, enc_r(OPCODE_OP_32, 0, 1, 2, 3, 3));

    // ---- OP_IMM_32 (0x1B) -- I-type / ITypeShamt (RV64 only) -----------
    test_i!(dispatch_addiw => Addiw, opcode = 0x1b, funct3 = 0, rd = 17, rs1 = 18, imm = 100);
    test_i_shamt5!(dispatch_slliw => Slliw, opcode = 0x1b, funct7 = 0,    funct3 = 1, rd = 19, rs1 = 20, shamt = 5);
    test_i_shamt5!(dispatch_srliw => Srliw, opcode = 0x1b, funct7 = 0,    funct3 = 5, rd = 21, rs1 = 22, shamt = 31);
    test_i_shamt5!(dispatch_sraiw => Sraiw, opcode = 0x1b, funct7 = 0x20, funct3 = 5, rd = 23, rs1 = 24, shamt = 0);
    // For W-form shifts the spec mandates shamt is 5 bits (0-31). Setting
    // shamt=32 is illegal even though the field syntactically allows it.
    test_rejects!(rejects_slliw_shamt_too_large,
        enc_i_shamt5(OPCODE_OP_IMM_32, 0, 32, 1, 1, 2));
    test_rejects!(rejects_srliw_shamt_too_large,
        enc_i_shamt5(OPCODE_OP_IMM_32, 0, 32, 1, 5, 2));
    // SLLIW with bit 26 set (funct6=1) is illegal -- the spec requires
    // funct7=0 for SLLIW, which means bits 31:25 are all zero.
    test_rejects!(rejects_slliw_invalid_funct6,
        enc_i_shamt5(OPCODE_OP_IMM_32, 0x02, 5, 1, 1, 2));
    // SRLIW/SRAIW with funct6 not in {0, 0x10} is illegal even when
    // shamt < 32.
    test_rejects!(rejects_srliw_invalid_funct6,
        enc_i_shamt5(OPCODE_OP_IMM_32, 0x02, 5, 1, 5, 2));
    // OP_IMM_32 has only ADDIW (funct3=0), SLLIW (1), SRLIW/SRAIW (5).
    // funct3 in {2, 3, 4, 6, 7} is unassigned.
    test_rejects!(rejects_op_imm_32_funct3_2, enc_i(OPCODE_OP_IMM_32, 0, 1, 2, 2));
    test_rejects!(rejects_op_imm_32_funct3_3, enc_i(OPCODE_OP_IMM_32, 0, 1, 3, 2));
    test_rejects!(rejects_op_imm_32_funct3_4, enc_i(OPCODE_OP_IMM_32, 0, 1, 4, 2));
    test_rejects!(rejects_op_imm_32_funct3_6, enc_i(OPCODE_OP_IMM_32, 0, 1, 6, 2));
    test_rejects!(rejects_op_imm_32_funct3_7, enc_i(OPCODE_OP_IMM_32, 0, 1, 7, 2));

    // ---- SYSTEM (0x73) -- ECALL/EBREAK/CSRs unsupported ----------------
    test_rejects!(rejects_ebreak,  0x00100073); // ebreak
    test_rejects!(rejects_csrrw,   0x30001073); // csrrw x0, mstatus, x0
}
