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
            0b001_0100 => Some(processor.process_bset(dec_insn)),
            0b010_0100 => Some(processor.process_bclr(dec_insn)),
            0b011_0000 => Some(processor.process_rol(dec_insn)),
            0b011_0100 => Some(processor.process_binv(dec_insn)),
            _ => None,
        },
        0b010 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_slt(dec_insn)),
            0b000_0001 => Some(processor.process_mulhsu(dec_insn)),
            0b001_0000 => Some(processor.process_sh1add(dec_insn)),
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
            0b000_0101 => Some(processor.process_min(dec_insn)),
            0b001_0000 => Some(processor.process_sh2add(dec_insn)),
            0b010_0000 => Some(processor.process_xnor(dec_insn)),
            _ => None,
        },
        0b101 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_srl(dec_insn)),
            0b000_0001 => Some(processor.process_divu(dec_insn)),
            0b000_0101 => Some(processor.process_minu(dec_insn)),
            0b010_0000 => Some(processor.process_sra(dec_insn)),
            0b010_0100 => Some(processor.process_bext(dec_insn)),
            0b011_0000 => Some(processor.process_ror(dec_insn)),
            _ => None,
        },
        0b110 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_or(dec_insn)),
            0b000_0001 => Some(processor.process_rem(dec_insn)),
            0b000_0101 => Some(processor.process_max(dec_insn)),
            0b001_0000 => Some(processor.process_sh3add(dec_insn)),
            0b010_0000 => Some(processor.process_orn(dec_insn)),
            _ => None,
        },
        0b111 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_and(dec_insn)),
            0b000_0001 => Some(processor.process_remu(dec_insn)),
            0b000_0101 => Some(processor.process_maxu(dec_insn)),
            0b010_0000 => Some(processor.process_andn(dec_insn)),
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
                0b001_010 => Some(processor.process_bseti(dec_insn_shamt)),
                0b010_010 => Some(processor.process_bclri(dec_insn_shamt)),
                0b011_010 => Some(processor.process_binvi(dec_insn_shamt)),
                // Unary Zbb ops: the sub-operation lives in the shamt field.
                0b011_000 => match dec_insn_shamt.shamt {
                    0b00000 => Some(processor.process_clz(dec_insn)),
                    0b00001 => Some(processor.process_ctz(dec_insn)),
                    0b00010 => Some(processor.process_cpop(dec_insn)),
                    0b00100 => Some(processor.process_sext_b(dec_insn)),
                    0b00101 => Some(processor.process_sext_h(dec_insn)),
                    _ => None,
                },
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
                0b010_010 => Some(processor.process_bexti(dec_insn_shamt)),
                0b011_000 => Some(processor.process_rori(dec_insn_shamt)),
                0b001_010 if dec_insn_shamt.shamt == 0b000_111 => {
                    Some(processor.process_orc_b(dec_insn))
                }
                0b011_010 if dec_insn_shamt.shamt == 0b111_000 => {
                    Some(processor.process_rev8(dec_insn))
                }
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
            match dec_insn_shamt.funct6 {
                0b000_000 if dec_insn_shamt.shamt < 32 => {
                    Some(processor.process_slliw(dec_insn_shamt))
                }
                // slli.uw takes the full 6-bit shamt (it shifts a zero-extended
                // 32-bit value within a 64-bit register).
                0b000_010 => Some(processor.process_slli_uw(dec_insn_shamt)),
                // Unary Zbb W-form ops: the sub-operation lives in the shamt field.
                0b011_000 => match dec_insn_shamt.shamt {
                    0b00000 => Some(processor.process_clzw(dec_insn)),
                    0b00001 => Some(processor.process_ctzw(dec_insn)),
                    0b00010 => Some(processor.process_cpopw(dec_insn)),
                    _ => None,
                },
                _ => None,
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
                0b011_000 => Some(processor.process_roriw(dec_insn_shamt)),
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
            0b000_0100 => Some(processor.process_add_uw(dec_insn)),
            0b010_0000 => Some(processor.process_subw(dec_insn)),
            _ => None,
        },
        0b001 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sllw(dec_insn)),
            0b011_0000 => Some(processor.process_rolw(dec_insn)),
            _ => None,
        },
        0b010 => match dec_insn.funct7 {
            0b001_0000 => Some(processor.process_sh1add_uw(dec_insn)),
            _ => None,
        },
        0b100 => match dec_insn.funct7 {
            0b000_0001 => Some(processor.process_divw(dec_insn)),
            // zext.h shares funct7 with a reserved encoding space; only rs2 = 0 is valid.
            0b000_0100 if dec_insn.rs2 == 0 => Some(processor.process_zext_h(dec_insn)),
            0b001_0000 => Some(processor.process_sh2add_uw(dec_insn)),
            _ => None,
        },
        0b101 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_srlw(dec_insn)),
            0b000_0001 => Some(processor.process_divuw(dec_insn)),
            0b010_0000 => Some(processor.process_sraw(dec_insn)),
            0b011_0000 => Some(processor.process_rorw(dec_insn)),
            _ => None,
        },
        0b110 => match dec_insn.funct7 {
            0b000_0001 => Some(processor.process_remw(dec_insn)),
            0b001_0000 => Some(processor.process_sh3add_uw(dec_insn)),
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

/// Returns whether `insn_bits` encodes a bit-manipulation (Zba/Zbb/Zbs) instruction.
///
/// This must stay in lockstep with the dispatch arms above (enforced by the
/// `bitmanip_predicate_matches_dispatch` test). Transpiler extensions use it to
/// decide which extension owns a word: the RV64IM extension skips exactly the
/// words the bit-manipulation extension claims, so registering both never
/// yields an ambiguous transpilation.
pub fn is_bitmanip_instruction(insn_bits: u32) -> bool {
    let opcode: u32 = insn_bits & 0x7f;

    match opcode {
        instruction_formats::OPCODE_OP => {
            let dec_insn = instruction_formats::RType::new(insn_bits);
            matches!(
                (dec_insn.funct3, dec_insn.funct7),
                (0b001, 0b001_0100)     // bset
                    | (0b001, 0b010_0100) // bclr
                    | (0b001, 0b011_0000) // rol
                    | (0b001, 0b011_0100) // binv
                    | (0b010, 0b001_0000) // sh1add
                    | (0b100, 0b000_0101) // min
                    | (0b100, 0b001_0000) // sh2add
                    | (0b100, 0b010_0000) // xnor
                    | (0b101, 0b000_0101) // minu
                    | (0b101, 0b010_0100) // bext
                    | (0b101, 0b011_0000) // ror
                    | (0b110, 0b000_0101) // max
                    | (0b110, 0b001_0000) // sh3add
                    | (0b110, 0b010_0000) // orn
                    | (0b111, 0b000_0101) // maxu
                    | (0b111, 0b010_0000) // andn
            )
        }
        instruction_formats::OPCODE_OP_32 => {
            let dec_insn = instruction_formats::RType::new(insn_bits);
            match (dec_insn.funct3, dec_insn.funct7) {
                (0b000, 0b000_0100) => true,              // add.uw
                (0b001, 0b011_0000) => true,              // rolw
                (0b010, 0b001_0000) => true,              // sh1add.uw
                (0b100, 0b000_0100) => dec_insn.rs2 == 0, // zext.h
                (0b100, 0b001_0000) => true,              // sh2add.uw
                (0b101, 0b011_0000) => true,              // rorw
                (0b110, 0b001_0000) => true,              // sh3add.uw
                _ => false,
            }
        }
        instruction_formats::OPCODE_OP_IMM => {
            let dec_insn = instruction_formats::ITypeShamt::new(insn_bits);
            match (dec_insn.funct3, dec_insn.funct6) {
                (0b001, 0b001_010) => true, // bseti
                (0b001, 0b010_010) => true, // bclri
                (0b001, 0b011_010) => true, // binvi
                // clz/ctz/cpop/sext.b/sext.h
                (0b001, 0b011_000) => {
                    matches!(dec_insn.shamt, 0b00000..=0b00010 | 0b00100 | 0b00101)
                }
                (0b101, 0b001_010) => dec_insn.shamt == 0b000_111, // orc.b
                (0b101, 0b010_010) => true,                        // bexti
                (0b101, 0b011_000) => true,                        // rori
                (0b101, 0b011_010) => dec_insn.shamt == 0b111_000, // rev8
                _ => false,
            }
        }
        instruction_formats::OPCODE_OP_IMM_32 => {
            let dec_insn = instruction_formats::ITypeShamt::new(insn_bits);
            match (dec_insn.funct3, dec_insn.funct6) {
                (0b001, 0b000_010) => true, // slli.uw
                // clzw/ctzw/cpopw
                (0b001, 0b011_000) => dec_insn.shamt <= 0b00010,
                (0b101, 0b011_000) => dec_insn.shamt < 32, // roriw
                _ => false,
            }
        }
        _ => false,
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

        // Zba (OPCODE_OP / OPCODE_OP_32 / OPCODE_OP_IMM_32)
        AddUw     process_add_uw     RType,
        Sh1add    process_sh1add     RType,
        Sh2add    process_sh2add     RType,
        Sh3add    process_sh3add     RType,
        Sh1addUw  process_sh1add_uw  RType,
        Sh2addUw  process_sh2add_uw  RType,
        Sh3addUw  process_sh3add_uw  RType,
        SlliUw    process_slli_uw    ITypeShamt,

        // Zbb: logical with negate (OPCODE_OP)
        Andn    process_andn    RType,
        Orn     process_orn     RType,
        Xnor    process_xnor    RType,

        // Zbb: counts (OPCODE_OP_IMM / OPCODE_OP_IMM_32, unary)
        Clz     process_clz     IType,
        Ctz     process_ctz     IType,
        Cpop    process_cpop    IType,
        Clzw    process_clzw    IType,
        Ctzw    process_ctzw    IType,
        Cpopw   process_cpopw   IType,

        // Zbb: min/max (OPCODE_OP)
        Min     process_min     RType,
        Minu    process_minu    RType,
        Max     process_max     RType,
        Maxu    process_maxu    RType,

        // Zbb: sign/zero extension
        SextB   process_sext_b  IType,
        SextH   process_sext_h  IType,
        ZextH   process_zext_h  RType,

        // Zbb: rotates
        Rol     process_rol     RType,
        Ror     process_ror     RType,
        Rori    process_rori    ITypeShamt,
        Rolw    process_rolw    RType,
        Rorw    process_rorw    RType,
        Roriw   process_roriw   ITypeShamt,

        // Zbb: byte ops (unary)
        OrcB    process_orc_b   IType,
        Rev8    process_rev8    IType,

        // Zbs (OPCODE_OP / OPCODE_OP_IMM)
        Bclr    process_bclr    RType,
        Bset    process_bset    RType,
        Binv    process_binv    RType,
        Bext    process_bext    RType,
        Bclri   process_bclri   ITypeShamt,
        Bseti   process_bseti   ITypeShamt,
        Binvi   process_binvi   ITypeShamt,
        Bexti   process_bexti   ITypeShamt,
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

    // ---- Bit-manipulation (Zba / Zbb / Zbs) dispatch --------------------

    #[test]
    fn dispatch_zba() {
        check_r_dispatch(OPCODE_OP_32, 0x04, 0, 1, 2, 3, Called::AddUw);
        check_r_dispatch(OPCODE_OP, 0x10, 2, 4, 5, 6, Called::Sh1add);
        check_r_dispatch(OPCODE_OP, 0x10, 4, 7, 8, 9, Called::Sh2add);
        check_r_dispatch(OPCODE_OP, 0x10, 6, 10, 11, 12, Called::Sh3add);
        check_r_dispatch(OPCODE_OP_32, 0x10, 2, 13, 14, 15, Called::Sh1addUw);
        check_r_dispatch(OPCODE_OP_32, 0x10, 4, 16, 17, 18, Called::Sh2addUw);
        check_r_dispatch(OPCODE_OP_32, 0x10, 6, 19, 20, 21, Called::Sh3addUw);
        // slli.uw takes the full 6-bit shamt, unlike slliw.
        check_i_shamt6_dispatch(OPCODE_OP_IMM_32, 0x02, 1, 22, 23, 63, Called::SlliUw);
        check_i_shamt6_dispatch(OPCODE_OP_IMM_32, 0x02, 1, 22, 23, 0, Called::SlliUw);
    }

    #[test]
    fn dispatch_zbb_logic_minmax_rotates() {
        check_r_dispatch(OPCODE_OP, 0x20, 7, 1, 2, 3, Called::Andn);
        check_r_dispatch(OPCODE_OP, 0x20, 6, 4, 5, 6, Called::Orn);
        check_r_dispatch(OPCODE_OP, 0x20, 4, 7, 8, 9, Called::Xnor);
        check_r_dispatch(OPCODE_OP, 0x05, 4, 10, 11, 12, Called::Min);
        check_r_dispatch(OPCODE_OP, 0x05, 5, 13, 14, 15, Called::Minu);
        check_r_dispatch(OPCODE_OP, 0x05, 6, 16, 17, 18, Called::Max);
        check_r_dispatch(OPCODE_OP, 0x05, 7, 19, 20, 21, Called::Maxu);
        check_r_dispatch(OPCODE_OP, 0x30, 1, 22, 23, 24, Called::Rol);
        check_r_dispatch(OPCODE_OP, 0x30, 5, 25, 26, 27, Called::Ror);
        check_r_dispatch(OPCODE_OP_32, 0x30, 1, 28, 29, 30, Called::Rolw);
        check_r_dispatch(OPCODE_OP_32, 0x30, 5, 31, 0, 1, Called::Rorw);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x18, 5, 2, 3, 63, Called::Rori);
        check_i_shamt5_dispatch(OPCODE_OP_IMM_32, 0x30, 5, 4, 5, 31, Called::Roriw);
    }

    #[test]
    fn dispatch_zbb_unary() {
        // Unary Zbb ops encode the sub-operation in the shamt/rs2 field, so
        // dispatch is on the full 12-bit immediate. The processor receives
        // the plain IType (imm carries the funct12 value).
        check_i_dispatch(OPCODE_OP_IMM, 1, 1, 2, 0x600, Called::Clz);
        check_i_dispatch(OPCODE_OP_IMM, 1, 3, 4, 0x601, Called::Ctz);
        check_i_dispatch(OPCODE_OP_IMM, 1, 5, 6, 0x602, Called::Cpop);
        check_i_dispatch(OPCODE_OP_IMM, 1, 7, 8, 0x604, Called::SextB);
        check_i_dispatch(OPCODE_OP_IMM, 1, 9, 10, 0x605, Called::SextH);
        check_i_dispatch(OPCODE_OP_IMM_32, 1, 11, 12, 0x600, Called::Clzw);
        check_i_dispatch(OPCODE_OP_IMM_32, 1, 13, 14, 0x601, Called::Ctzw);
        check_i_dispatch(OPCODE_OP_IMM_32, 1, 15, 16, 0x602, Called::Cpopw);
        check_i_dispatch(OPCODE_OP_IMM, 5, 17, 18, 0x287, Called::OrcB);
        check_i_dispatch(OPCODE_OP_IMM, 5, 19, 20, 0x6b8, Called::Rev8);
        // zext.h is R-type on OPCODE_OP_32 with rs2 hardwired to 0.
        check_r_dispatch(OPCODE_OP_32, 0x04, 4, 21, 22, 0, Called::ZextH);
    }

    #[test]
    fn dispatch_zbs() {
        check_r_dispatch(OPCODE_OP, 0x24, 1, 1, 2, 3, Called::Bclr);
        check_r_dispatch(OPCODE_OP, 0x14, 1, 4, 5, 6, Called::Bset);
        check_r_dispatch(OPCODE_OP, 0x34, 1, 7, 8, 9, Called::Binv);
        check_r_dispatch(OPCODE_OP, 0x24, 5, 10, 11, 12, Called::Bext);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x12, 1, 13, 14, 63, Called::Bclri);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x0a, 1, 15, 16, 0, Called::Bseti);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x1a, 1, 17, 18, 32, Called::Binvi);
        check_i_shamt6_dispatch(OPCODE_OP_IMM, 0x12, 5, 19, 20, 50, Called::Bexti);
    }

    /// Golden encodings generated with
    /// `llvm-mc -triple=riscv64 -mattr=+zba,+zbb,+zbs -show-encoding` (LLVM 22),
    /// i.e. the exact words the guest compiler emits. Pins both the funct-space
    /// mapping and the operand field extraction for every B-ext instruction.
    #[test]
    fn dispatch_bitmanip_llvm_golden_words() {
        #[rustfmt::skip]
        let cases: &[(u32, Called)] = &[
            // add.uw x5, x6, x7
            (0x087302bb, Called::AddUw(RType { funct7: 0x04, rs2: 7, rs1: 6, funct3: 0, rd: 5 })),
            // sh1add x8, x9, x10
            (0x20a4a433, Called::Sh1add(RType { funct7: 0x10, rs2: 10, rs1: 9, funct3: 2, rd: 8 })),
            // sh2add x11, x12, x13
            (0x20d645b3, Called::Sh2add(RType { funct7: 0x10, rs2: 13, rs1: 12, funct3: 4, rd: 11 })),
            // sh3add x14, x15, x16
            (0x2107e733, Called::Sh3add(RType { funct7: 0x10, rs2: 16, rs1: 15, funct3: 6, rd: 14 })),
            // sh1add.uw x17, x18, x19
            (0x213928bb, Called::Sh1addUw(RType { funct7: 0x10, rs2: 19, rs1: 18, funct3: 2, rd: 17 })),
            // sh2add.uw x20, x21, x22
            (0x216aca3b, Called::Sh2addUw(RType { funct7: 0x10, rs2: 22, rs1: 21, funct3: 4, rd: 20 })),
            // sh3add.uw x23, x24, x25
            (0x219c6bbb, Called::Sh3addUw(RType { funct7: 0x10, rs2: 25, rs1: 24, funct3: 6, rd: 23 })),
            // slli.uw x26, x27, 37
            (0x0a5d9d1b, Called::SlliUw(ITypeShamt { funct6: 0x02, shamt: 37, rs1: 27, funct3: 1, rd: 26 })),
            // andn x5, x6, x7
            (0x407372b3, Called::Andn(RType { funct7: 0x20, rs2: 7, rs1: 6, funct3: 7, rd: 5 })),
            // orn x8, x9, x10
            (0x40a4e433, Called::Orn(RType { funct7: 0x20, rs2: 10, rs1: 9, funct3: 6, rd: 8 })),
            // xnor x11, x12, x13
            (0x40d645b3, Called::Xnor(RType { funct7: 0x20, rs2: 13, rs1: 12, funct3: 4, rd: 11 })),
            // clz x14, x15
            (0x60079713, Called::Clz(IType { imm: 0x600, rs1: 15, funct3: 1, rd: 14 })),
            // ctz x16, x17
            (0x60189813, Called::Ctz(IType { imm: 0x601, rs1: 17, funct3: 1, rd: 16 })),
            // cpop x18, x19
            (0x60299913, Called::Cpop(IType { imm: 0x602, rs1: 19, funct3: 1, rd: 18 })),
            // clzw x20, x21
            (0x600a9a1b, Called::Clzw(IType { imm: 0x600, rs1: 21, funct3: 1, rd: 20 })),
            // ctzw x22, x23
            (0x601b9b1b, Called::Ctzw(IType { imm: 0x601, rs1: 23, funct3: 1, rd: 22 })),
            // cpopw x24, x25
            (0x602c9c1b, Called::Cpopw(IType { imm: 0x602, rs1: 25, funct3: 1, rd: 24 })),
            // min x5, x6, x7
            (0x0a7342b3, Called::Min(RType { funct7: 0x05, rs2: 7, rs1: 6, funct3: 4, rd: 5 })),
            // minu x8, x9, x10
            (0x0aa4d433, Called::Minu(RType { funct7: 0x05, rs2: 10, rs1: 9, funct3: 5, rd: 8 })),
            // max x11, x12, x13
            (0x0ad665b3, Called::Max(RType { funct7: 0x05, rs2: 13, rs1: 12, funct3: 6, rd: 11 })),
            // maxu x14, x15, x16
            (0x0b07f733, Called::Maxu(RType { funct7: 0x05, rs2: 16, rs1: 15, funct3: 7, rd: 14 })),
            // sext.b x17, x18
            (0x60491893, Called::SextB(IType { imm: 0x604, rs1: 18, funct3: 1, rd: 17 })),
            // sext.h x19, x20
            (0x605a1993, Called::SextH(IType { imm: 0x605, rs1: 20, funct3: 1, rd: 19 })),
            // zext.h x21, x22
            (0x080b4abb, Called::ZextH(RType { funct7: 0x04, rs2: 0, rs1: 22, funct3: 4, rd: 21 })),
            // rol x5, x6, x7
            (0x607312b3, Called::Rol(RType { funct7: 0x30, rs2: 7, rs1: 6, funct3: 1, rd: 5 })),
            // ror x8, x9, x10
            (0x60a4d433, Called::Ror(RType { funct7: 0x30, rs2: 10, rs1: 9, funct3: 5, rd: 8 })),
            // rori x11, x12, 45
            (0x62d65593, Called::Rori(ITypeShamt { funct6: 0x18, shamt: 45, rs1: 12, funct3: 5, rd: 11 })),
            // rolw x13, x14, x15
            (0x60f716bb, Called::Rolw(RType { funct7: 0x30, rs2: 15, rs1: 14, funct3: 1, rd: 13 })),
            // rorw x16, x17, x18
            (0x6128d83b, Called::Rorw(RType { funct7: 0x30, rs2: 18, rs1: 17, funct3: 5, rd: 16 })),
            // roriw x19, x20, 21
            (0x615a599b, Called::Roriw(ITypeShamt { funct6: 0x18, shamt: 21, rs1: 20, funct3: 5, rd: 19 })),
            // orc.b x22, x23
            (0x287bdb13, Called::OrcB(IType { imm: 0x287, rs1: 23, funct3: 5, rd: 22 })),
            // rev8 x24, x25
            (0x6b8cdc13, Called::Rev8(IType { imm: 0x6b8, rs1: 25, funct3: 5, rd: 24 })),
            // bclr x5, x6, x7
            (0x487312b3, Called::Bclr(RType { funct7: 0x24, rs2: 7, rs1: 6, funct3: 1, rd: 5 })),
            // bset x8, x9, x10
            (0x28a49433, Called::Bset(RType { funct7: 0x14, rs2: 10, rs1: 9, funct3: 1, rd: 8 })),
            // binv x11, x12, x13
            (0x68d615b3, Called::Binv(RType { funct7: 0x34, rs2: 13, rs1: 12, funct3: 1, rd: 11 })),
            // bext x14, x15, x16
            (0x4907d733, Called::Bext(RType { funct7: 0x24, rs2: 16, rs1: 15, funct3: 5, rd: 14 })),
            // bclri x17, x18, 47
            (0x4af91893, Called::Bclri(ITypeShamt { funct6: 0x12, shamt: 47, rs1: 18, funct3: 1, rd: 17 })),
            // bseti x19, x20, 48
            (0x2b0a1993, Called::Bseti(ITypeShamt { funct6: 0x0a, shamt: 48, rs1: 20, funct3: 1, rd: 19 })),
            // binvi x21, x22, 49
            (0x6b1b1a93, Called::Binvi(ITypeShamt { funct6: 0x1a, shamt: 49, rs1: 22, funct3: 1, rd: 21 })),
            // bexti x23, x24, 50
            (0x4b2c5b93, Called::Bexti(ITypeShamt { funct6: 0x12, shamt: 50, rs1: 24, funct3: 5, rd: 23 })),
        ];
        for (bits, expected) in cases {
            assert_eq!(
                process_instruction(&mut Recorder, *bits).as_ref(),
                Some(expected),
                "golden word {bits:#010x} did not dispatch as expected"
            );
            assert!(
                is_bitmanip_instruction(*bits),
                "golden word {bits:#010x} not classified as bit-manipulation"
            );
        }
    }

    #[test]
    fn rejects_bitmanip_reserved_encodings() {
        // Reserved shamt values in the unary Zbb funct12 space (only
        // 0..=2, 4, 5 are defined under funct6 = 0b011000, funct3 = 001).
        check_rejects(enc_i(OPCODE_OP_IMM, 0x603, 1, 1, 2));
        check_rejects(enc_i(OPCODE_OP_IMM, 0x606, 1, 1, 2));
        check_rejects(enc_i(OPCODE_OP_IMM_32, 0x603, 1, 1, 2));
        check_rejects(enc_i(OPCODE_OP_IMM_32, 0x604, 1, 1, 2)); // no sext.b W-form
                                                                // orc.b / rev8 are single points in their funct6 space, not ranges.
        check_rejects(enc_i(OPCODE_OP_IMM, 0x286, 1, 5, 2)); // orc.b low bits - 1
        check_rejects(enc_i(OPCODE_OP_IMM, 0x6b9, 1, 5, 2)); // rev8 low bits + 1
                                                             // zext.h requires rs2 = 0 (nonzero rs2 would be packw, unsupported).
        check_rejects(enc_r(OPCODE_OP_32, 0x04, 1, 2, 4, 3));
        // roriw is a 5-bit shamt; bit 25 set (shamt >= 32) is reserved.
        check_rejects(enc_i_shamt6(OPCODE_OP_IMM_32, 0x18, 32, 1, 5, 2));
        // No W-forms exist for min/max, single-bit ops, or logic-with-negate.
        check_rejects(enc_r(OPCODE_OP_32, 0x05, 1, 2, 4, 3));
        check_rejects(enc_r(OPCODE_OP_32, 0x24, 1, 2, 1, 3));
        check_rejects(enc_r(OPCODE_OP_32, 0x20, 1, 2, 7, 3));
    }

    /// `is_bitmanip_instruction` must classify a word as bit-manipulation
    /// exactly when dispatch lands on a B-ext processor method. Sweeps the
    /// whole funct space of the four affected major opcodes.
    #[test]
    fn bitmanip_predicate_matches_dispatch() {
        fn is_bitmanip_called(called: &Called) -> bool {
            matches!(
                called,
                Called::AddUw(_)
                    | Called::Sh1add(_)
                    | Called::Sh2add(_)
                    | Called::Sh3add(_)
                    | Called::Sh1addUw(_)
                    | Called::Sh2addUw(_)
                    | Called::Sh3addUw(_)
                    | Called::SlliUw(_)
                    | Called::Andn(_)
                    | Called::Orn(_)
                    | Called::Xnor(_)
                    | Called::Clz(_)
                    | Called::Ctz(_)
                    | Called::Cpop(_)
                    | Called::Clzw(_)
                    | Called::Ctzw(_)
                    | Called::Cpopw(_)
                    | Called::Min(_)
                    | Called::Minu(_)
                    | Called::Max(_)
                    | Called::Maxu(_)
                    | Called::SextB(_)
                    | Called::SextH(_)
                    | Called::ZextH(_)
                    | Called::Rol(_)
                    | Called::Ror(_)
                    | Called::Rori(_)
                    | Called::Rolw(_)
                    | Called::Rorw(_)
                    | Called::Roriw(_)
                    | Called::OrcB(_)
                    | Called::Rev8(_)
                    | Called::Bclr(_)
                    | Called::Bset(_)
                    | Called::Binv(_)
                    | Called::Bext(_)
                    | Called::Bclri(_)
                    | Called::Bseti(_)
                    | Called::Binvi(_)
                    | Called::Bexti(_)
            )
        }

        let check = |bits: u32| {
            let dispatched_bitmanip = process_instruction(&mut Recorder, bits)
                .as_ref()
                .is_some_and(is_bitmanip_called);
            assert_eq!(
                is_bitmanip_instruction(bits),
                dispatched_bitmanip,
                "predicate/dispatch mismatch for word {bits:#010x}"
            );
        };

        // R-type space: all funct3 x funct7, with rs2 = 0 as well to cover the
        // zext.h special case.
        for opcode in [OPCODE_OP, OPCODE_OP_32] {
            for funct3 in 0..8 {
                for funct7 in 0..128 {
                    for rs2 in [0, 3] {
                        check(enc_r(opcode, funct7, rs2, 2, funct3, 1));
                    }
                }
            }
        }
        // I-type space: all funct3 x full 12-bit immediate (covers every
        // funct6/shamt split and the unary funct12 points).
        for opcode in [OPCODE_OP_IMM, OPCODE_OP_IMM_32] {
            for funct3 in 0..8 {
                for imm12 in 0..4096 {
                    check(enc_i(opcode, imm12, 2, funct3, 1));
                }
            }
        }
        // Other major opcodes are never bit-manipulation.
        for opcode in [
            OPCODE_LOAD,
            OPCODE_STORE,
            OPCODE_BRANCH,
            OPCODE_JAL,
            OPCODE_JALR,
            OPCODE_LUI,
            OPCODE_AUIPC,
            OPCODE_MISC_MEM,
        ] {
            check(enc_i(opcode, 0, 2, 0, 1));
        }
    }
}
