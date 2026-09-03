//! Shared encoder helpers for tests.
//!
//! Each function constructs the canonical 32-bit instruction word for its
//! format, per *The RISC-V Instruction Set Manual, Volume I: Unprivileged
//! Architecture*, Version 20260120 (Official Release), §2.2 (Base Instruction
//! Formats) and §2.3 (Immediate Encoding Variants).

pub(crate) fn enc_r(opcode: u32, funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

pub(crate) fn enc_i(opcode: u32, imm: i32, rs1: u32, funct3: u32, rd: u32) -> u32 {
    let imm_u = (imm as u32) & 0xfff;
    (imm_u << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// RV64 OP_IMM shift form: 6-bit `funct6` in bits 31:26, 6-bit `shamt` in bits 25:20.
pub(crate) fn enc_i_shamt6(
    opcode: u32,
    funct6: u32,
    shamt: u32,
    rs1: u32,
    funct3: u32,
    rd: u32,
) -> u32 {
    (funct6 << 26) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// RV64 OP_IMM_32 (W-form) shift form: 7-bit `funct7` in bits 31:25, 5-bit `shamt` in bits 24:20.
/// Bit 25 must be 0 (encoded via `funct7`'s low bit being 0).
pub(crate) fn enc_i_shamt5(
    opcode: u32,
    funct7: u32,
    shamt: u32,
    rs1: u32,
    funct3: u32,
    rd: u32,
) -> u32 {
    (funct7 << 25) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

pub(crate) fn enc_s(opcode: u32, imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm_u = (imm as u32) & 0xfff;
    let imm_hi = (imm_u >> 5) & 0x7f;
    let imm_lo = imm_u & 0x1f;
    (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode
}

pub(crate) fn enc_b(opcode: u32, imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
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

pub(crate) fn enc_u(opcode: u32, imm: i32, rd: u32) -> u32 {
    ((imm as u32) & 0xffff_f000) | (rd << 7) | opcode
}

pub(crate) fn enc_j(opcode: u32, imm: i32, rd: u32) -> u32 {
    let imm_u = (imm as u32) & 0x1f_ffff;
    let bit20 = (imm_u >> 20) & 1;
    let bits_10_1 = (imm_u >> 1) & 0x3ff;
    let bit11 = (imm_u >> 11) & 1;
    let bits_19_12 = (imm_u >> 12) & 0xff;
    (bit20 << 31) | (bits_10_1 << 21) | (bit11 << 20) | (bits_19_12 << 12) | (rd << 7) | opcode
}
