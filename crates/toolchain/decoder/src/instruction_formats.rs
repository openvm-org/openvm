// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Structures and constants for instruction decoding
//!
//! The structures directly relate to the RISC-V instruction formats described in the
//! specification. See the [RISC-V specification](https://riscv.org/technical/specifications/) for
//! further details

pub const OPCODE_LOAD: u32 = 0x03;
pub const OPCODE_MISC_MEM: u32 = 0x0f;
pub const OPCODE_OP_IMM: u32 = 0x13;
pub const OPCODE_AUIPC: u32 = 0x17;
pub const OPCODE_OP_IMM_32: u32 = 0x1b;
pub const OPCODE_STORE: u32 = 0x23;
pub const OPCODE_OP: u32 = 0x33;
pub const OPCODE_LUI: u32 = 0x37;
pub const OPCODE_OP_32: u32 = 0x3b;
pub const OPCODE_BRANCH: u32 = 0x63;
pub const OPCODE_JALR: u32 = 0x67;
pub const OPCODE_JAL: u32 = 0x6f;

#[derive(Debug, PartialEq)]
pub struct RType {
    pub funct7: u32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl RType {
    pub fn new(insn: u32) -> RType {
        RType {
            funct7: (insn >> 25) & 0x7f,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct IType {
    pub imm: i32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl IType {
    pub fn new(insn: u32) -> IType {
        let uimm: i32 = ((insn >> 20) & 0x7ff) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };

        IType {
            imm,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ITypeShamt {
    pub funct6: u32,
    pub shamt: u32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl ITypeShamt {
    pub fn new(insn: u32) -> ITypeShamt {
        let itype = IType::new(insn);
        let shamt = (itype.imm as u32) & 0x3f;

        ITypeShamt {
            funct6: (insn >> 26) & 0x3f,
            shamt,
            rs1: itype.rs1,
            funct3: itype.funct3,
            rd: itype.rd,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl SType {
    pub fn new(insn: u32) -> SType {
        let uimm: i32 = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1f)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };

        SType {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl BType {
    pub fn new(insn: u32) -> BType {
        let uimm: i32 =
            (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1e) | ((insn & 0x80) << 4)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 12)
        } else {
            uimm
        };

        BType {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct UType {
    pub imm: i32,
    pub rd: usize,
}

impl UType {
    pub fn new(insn: u32) -> UType {
        UType {
            imm: (insn & 0xffff_f000) as i32,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct JType {
    pub imm: i32,
    pub rd: usize,
}

impl JType {
    pub fn new(insn: u32) -> JType {
        let uimm: i32 =
            ((insn & 0xff000) | ((insn & 0x100000) >> 9) | ((insn >> 20) & 0x7fe)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 20)
        } else {
            uimm
        };

        JType {
            imm,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtype() {
        assert_eq!(
            RType::new(0x0),
            RType {
                funct7: 0,
                rs2: 0,
                rs1: 0,
                funct3: 0,
                rd: 0
            }
        )
    }

    #[test]
    fn test_itype() {
        // addi x23, x31, 2047
        assert_eq!(
            IType::new(0x7fff8b93),
            IType {
                imm: 2047,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -1
        assert_eq!(
            IType::new(0xffff8b93),
            IType {
                imm: -1,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -2
        assert_eq!(
            IType::new(0xffef8b93),
            IType {
                imm: -2,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // ori x13, x7, 4
        assert_eq!(
            IType::new(0x8003e693),
            IType {
                imm: -2048,
                rs1: 7,
                funct3: 0b110,
                rd: 13
            }
        );
    }

    #[test]
    fn test_itype_shamt() {
        // slli x12, x5, 13
        assert_eq!(
            ITypeShamt::new(0x00d29613),
            ITypeShamt {
                funct6: 0,
                shamt: 13,
                rs1: 5,
                funct3: 0b001,
                rd: 12
            }
        );

        // srli x30, x19, 31
        assert_eq!(
            ITypeShamt::new(0x01f9df13),
            ITypeShamt {
                funct6: 0,
                shamt: 31,
                rs1: 19,
                funct3: 0b101,
                rd: 30
            }
        );

        // srai x7, x23, 0
        assert_eq!(
            ITypeShamt::new(0x400bd393),
            ITypeShamt {
                funct6: 0b010000,
                shamt: 0,
                rs1: 23,
                funct3: 0b101,
                rd: 7
            }
        );
    }

    #[test]
    fn test_stype() {
        // sb x31, -2048(x15)
        assert_eq!(
            SType::new(0x81f78023),
            SType {
                imm: -2048,
                rs2: 31,
                rs1: 15,
                funct3: 0,
            }
        );

        // sh x18, 2047(x3)
        assert_eq!(
            SType::new(0x7f219fa3),
            SType {
                imm: 2047,
                rs2: 18,
                rs1: 3,
                funct3: 1,
            }
        );

        // sw x8, 1(x23)
        assert_eq!(
            SType::new(0x008ba0a3),
            SType {
                imm: 1,
                rs2: 8,
                rs1: 23,
                funct3: 2,
            }
        );

        // sw x5, -1(x25)
        assert_eq!(
            SType::new(0xfe5cafa3),
            SType {
                imm: -1,
                rs2: 5,
                rs1: 25,
                funct3: 2,
            }
        );

        // sw x13, 7(x12)
        assert_eq!(
            SType::new(0x00d623a3),
            SType {
                imm: 7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );

        // sw x13, -7(x12)
        assert_eq!(
            SType::new(0xfed62ca3),
            SType {
                imm: -7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );
    }

    #[test]
    fn test_btype() {
        // beq x10, x14, .-4096
        assert_eq!(
            BType::new(0x80e50063),
            BType {
                imm: -4096,
                rs1: 10,
                rs2: 14,
                funct3: 0b000
            }
        );

        // blt x3, x21, .+4094
        assert_eq!(
            BType::new(0x7f51cfe3),
            BType {
                imm: 4094,
                rs1: 3,
                rs2: 21,
                funct3: 0b100
            }
        );

        // bge x18, x0, .-2
        assert_eq!(
            BType::new(0xfe095fe3),
            BType {
                imm: -2,
                rs1: 18,
                rs2: 0,
                funct3: 0b101
            }
        );

        // bne x15, x16, .+2
        assert_eq!(
            BType::new(0x01079163),
            BType {
                imm: 2,
                rs1: 15,
                rs2: 16,
                funct3: 0b001
            }
        );

        // bgeu x31, x8, .+18
        assert_eq!(
            BType::new(0x008ff963),
            BType {
                imm: 18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );

        // bgeu x31, x8, .-18
        assert_eq!(
            BType::new(0xfe8ff7e3),
            BType {
                imm: -18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );
    }

    #[test]
    fn test_utype() {
        // lui x0, 0xfffff
        assert_eq!(
            UType::new(0xfffff037),
            UType {
                imm: 0xfffff000_u32 as i32,
                rd: 0,
            }
        );

        // lui x31, 0x0
        assert_eq!(UType::new(0x00000fb7), UType { imm: 0x0, rd: 31 });

        // lui x17, 0x123ab
        assert_eq!(
            UType::new(0x123ab8b7),
            UType {
                imm: 0x123ab000,
                rd: 17,
            }
        );
    }

    #[test]
    fn test_jtype() {
        // jal x0, .+0xffffe
        assert_eq!(
            JType::new(0x7ffff06f),
            JType {
                imm: 0xffffe,
                rd: 0,
            }
        );

        // jal x31, .-0x100000
        assert_eq!(
            JType::new(0x80000fef),
            JType {
                imm: -0x100000,
                rd: 31,
            }
        );

        // jal x13, .-2
        assert_eq!(JType::new(0xfffff6ef), JType { imm: -2, rd: 13 });

        // jal x13, .+2
        assert_eq!(JType::new(0x002006ef), JType { imm: 2, rd: 13 });

        // jal x26, .-46
        assert_eq!(JType::new(0xfd3ffd6f), JType { imm: -46, rd: 26 });

        // jal x26, .+46
        assert_eq!(JType::new(0x02e00d6f), JType { imm: 46, rd: 26 });
    }

    // ---- RV64 edge cases and field-extraction stress -----------
    //
    // Section references in the tests below (e.g. "spec §4.2.1") refer to
    // *The RISC-V Instruction Set Manual, Volume I: Unprivileged
    // Architecture*, Version 20260120 (Official Release).
    //
    // Encoder helpers per spec §2.2 (Base Instruction Formats).

    fn enc_r(opcode: u32, funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn enc_i(opcode: u32, imm: i32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        let imm_u = (imm as u32) & 0xfff;
        (imm_u << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn enc_i_shamt6(opcode: u32, funct6: u32, shamt: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct6 << 26) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
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

    // ---- ITypeShamt RV64 6-bit shamt boundaries (spec §4.2.1) ----

    #[test]
    fn itype_shamt_rv64_shamt_32() {
        // shamt=32 (bit 25 set in the instruction word) is legal in RV64
        // OP_IMM shifts. Decoder must treat shamt as 6 bits, not 5; if it
        // were masking to 5 bits, this would decode as shamt=0.
        let bits = enc_i_shamt6(OPCODE_OP_IMM, 0, 32, 1, 1, 2);
        assert_eq!(
            ITypeShamt::new(bits),
            ITypeShamt {
                funct6: 0,
                shamt: 32,
                rs1: 1,
                funct3: 1,
                rd: 2,
            }
        );
    }

    #[test]
    fn itype_shamt_rv64_shamt_63() {
        // Max 6-bit shamt = 63 = 0x3f. All 6 shamt bits set.
        let bits = enc_i_shamt6(OPCODE_OP_IMM, 0, 63, 3, 5, 4);
        assert_eq!(
            ITypeShamt::new(bits),
            ITypeShamt {
                funct6: 0,
                shamt: 63,
                rs1: 3,
                funct3: 5,
                rd: 4,
            }
        );
    }

    #[test]
    fn itype_shamt_funct6_separation_from_shamt() {
        // funct6=0x10 (SRAI marker) + shamt=32 simultaneously. If the
        // decoder confused funct6 and shamt bit-ranges, funct6 would
        // bleed into shamt or vice versa. Independent fields:
        //   funct6 = bits[31:26]  → 0x10
        //   shamt  = bits[25:20]  → 32
        let bits = enc_i_shamt6(OPCODE_OP_IMM, 0x10, 32, 5, 5, 6);
        assert_eq!(
            ITypeShamt::new(bits),
            ITypeShamt {
                funct6: 0x10,
                shamt: 32,
                rs1: 5,
                funct3: 5,
                rd: 6,
            }
        );
    }

    #[test]
    fn itype_shamt_max_funct6() {
        // funct6=0x3f (all 6 bits set), shamt=0. The decoder's IType::new
        // sign-extends bits[31:20] -- here bits 31, 30, 29, 28, 27, 26 are
        // set, so the underlying IType.imm goes negative. ITypeShamt then
        // masks (imm as u32) & 0x3f to get shamt; this must yield 0
        // even though imm itself is negative.
        let bits = enc_i_shamt6(OPCODE_OP_IMM, 0x3f, 0, 1, 1, 2);
        assert_eq!(
            ITypeShamt::new(bits),
            ITypeShamt {
                funct6: 0x3f,
                shamt: 0,
                rs1: 1,
                funct3: 1,
                rd: 2,
            }
        );
    }

    // ---- 5-bit register-field masks (rd / rs1 / rs2) -------------------
    // Exercise the high boundary (x31) in each format.

    #[test]
    fn rtype_all_registers_31() {
        let bits = enc_r(OPCODE_OP, 0, 31, 31, 0, 31);
        assert_eq!(
            RType::new(bits),
            RType {
                funct7: 0,
                rs2: 31,
                rs1: 31,
                funct3: 0,
                rd: 31,
            }
        );
    }

    #[test]
    fn itype_registers_31() {
        let bits = enc_i(OPCODE_OP_IMM, 0, 31, 0, 31);
        assert_eq!(
            IType::new(bits),
            IType {
                imm: 0,
                rs1: 31,
                funct3: 0,
                rd: 31,
            }
        );
    }

    #[test]
    fn stype_registers_31() {
        let bits = enc_s(OPCODE_STORE, 0, 31, 31, 0);
        assert_eq!(
            SType::new(bits),
            SType {
                imm: 0,
                rs2: 31,
                rs1: 31,
                funct3: 0,
            }
        );
    }

    #[test]
    fn btype_registers_31() {
        let bits = enc_b(OPCODE_BRANCH, 0, 31, 31, 0);
        assert_eq!(
            BType::new(bits),
            BType {
                imm: 0,
                rs2: 31,
                rs1: 31,
                funct3: 0,
            }
        );
    }

    #[test]
    fn utype_rd_31() {
        let bits = enc_u(OPCODE_LUI, 0, 31);
        assert_eq!(UType::new(bits), UType { imm: 0, rd: 31 });
    }

    #[test]
    fn jtype_rd_31() {
        let bits = enc_j(OPCODE_JAL, 0, 31);
        assert_eq!(JType::new(bits), JType { imm: 0, rd: 31 });
    }

    // ---- One-hot immediate-bit sweeps ----------------------------------
    // S-type, B-type, and J-type immediates are scattered across the
    // instruction word in non-contiguous fields. For each bit position in
    // the decoded immediate, set ONLY that bit and confirm it round-trips
    // correctly. This is the single best check that the scattering
    // formulas are right.

    #[test]
    fn stype_one_hot_imm_bits() {
        // S-type imm is 12-bit signed. Bits 0..=10 are positive single-bit
        // values; bit 11 is the sign bit (= -2048 when set alone).
        for bit in 0..=10 {
            let imm = 1i32 << bit;
            let bits = enc_s(OPCODE_STORE, imm, 1, 2, 0);
            assert_eq!(
                SType::new(bits).imm,
                imm,
                "S-type imm bit {bit} did not round-trip"
            );
        }
        // Bit 11 set alone -> -2048.
        let bits = enc_s(OPCODE_STORE, -2048, 1, 2, 0);
        assert_eq!(SType::new(bits).imm, -2048);
    }

    #[test]
    fn btype_one_hot_imm_bits() {
        // B-type imm is 13-bit signed; imm[0] is always 0 (branch targets
        // are 2-byte aligned). So bits 1..=11 are positive; bit 12 is sign.
        for bit in 1..=11 {
            let imm = 1i32 << bit;
            let bits = enc_b(OPCODE_BRANCH, imm, 1, 2, 0);
            assert_eq!(
                BType::new(bits).imm,
                imm,
                "B-type imm bit {bit} did not round-trip"
            );
        }
        // Bit 12 set alone -> -4096.
        let bits = enc_b(OPCODE_BRANCH, -4096, 1, 2, 0);
        assert_eq!(BType::new(bits).imm, -4096);
    }

    #[test]
    fn jtype_one_hot_imm_bits() {
        // J-type imm is 21-bit signed; imm[0] is always 0 (jump targets
        // are 2-byte aligned). Bits 1..=19 are positive; bit 20 is sign.
        for bit in 1..=19 {
            let imm = 1i32 << bit;
            let bits = enc_j(OPCODE_JAL, imm, 0);
            assert_eq!(
                JType::new(bits).imm,
                imm,
                "J-type imm bit {bit} did not round-trip"
            );
        }
        // Bit 20 set alone -> -(1 << 20) = -1048576.
        let bits = enc_j(OPCODE_JAL, -(1 << 20), 0);
        assert_eq!(JType::new(bits).imm, -(1 << 20));
    }

    // ---- I-type sign-extension boundaries ------------------------------
    // Imm = 0 and +1 boundaries.

    #[test]
    fn itype_imm_zero() {
        let bits = enc_i(OPCODE_OP_IMM, 0, 0, 0, 0);
        assert_eq!(IType::new(bits).imm, 0);
    }

    #[test]
    fn itype_imm_plus_one() {
        let bits = enc_i(OPCODE_OP_IMM, 1, 0, 0, 0);
        assert_eq!(IType::new(bits).imm, 1);
    }

    // ---- U-type sign / low-bits handling -------------------------------

    #[test]
    fn utype_sign_bit_preserved() {
        // High bit of the 20-bit imm field set -> imm is negative i32.
        let bits = enc_u(OPCODE_LUI, 0x8000_0000_u32 as i32, 0);
        assert_eq!(UType::new(bits).imm, 0x8000_0000_u32 as i32);
    }

    #[test]
    fn utype_low_12_bits_zeroed() {
        // Even if the raw instruction word has bits set in positions 0..11
        // (which would be the rd/opcode fields), the decoded U-type imm
        // must have those bits zero -- only bits 12..31 of the instruction
        // contribute to imm.
        //
        // 0x12345abc: imm portion (bits 31..12) = 0x12345, rd (bits 11..7)
        // = (0xabc >> 7) & 0x1f. With low bits used as rd/opcode, the
        // decoded imm must still be exactly 0x12345000.
        let bits = 0x12345abc;
        let decoded = UType::new(bits);
        assert_eq!(decoded.imm, 0x1234_5000_u32 as i32);
    }
}
