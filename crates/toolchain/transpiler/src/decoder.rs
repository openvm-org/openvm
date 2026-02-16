// Portions of this file are derived from rrs-lib (https://github.com/GregAC/rrs)
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modifications:
// - Added RV64 support: 6-bit shamt in ITypeShamt, LD/SD/LWU opcodes,
//   OP-32/OP-IMM-32 decode, and corresponding InstructionProcessor methods
// - Added enum-based InstructionKind decoder as an alternative to the trait-based dispatch
// - Removed ITypeCSR (unused in OpenVM transpilers)
// - Removed system/CSR instruction decoding (handled separately by transpiler extensions)

//! RISC-V instruction decoder with RV32IM and RV64IM support.
//!
//! Provides:
//! - Instruction format structs for bitfield extraction
//! - [`InstructionProcessor`] trait + [`process_instruction`] for callback-based dispatch
//!   (compatible with the rrs-lib pattern, extended with RV64 methods)
//! - [`InstructionKind`] enum + [`decode`] for enum-based decoding

// ── Opcode constants ──────────────────────────────────────────────────────────

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
pub const OPCODE_SYSTEM: u32 = 0x73;

// ── Instruction format structs ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RType {
    pub funct7: u32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl RType {
    pub fn new(insn: u32) -> Self {
        Self {
            funct7: (insn >> 25) & 0x7f,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IType {
    pub imm: i32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl IType {
    pub fn new(insn: u32) -> Self {
        let uimm = ((insn >> 20) & 0x7ff) as i32;
        let imm = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };
        Self {
            imm,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

/// Shift-immediate format. Extracts a 6-bit shamt for RV64 compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ITypeShamt {
    pub funct6: u32,
    pub shamt: u32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl ITypeShamt {
    pub fn new(insn: u32) -> Self {
        Self {
            funct6: (insn >> 26) & 0x3f,
            shamt: (insn >> 20) & 0x3f,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl SType {
    pub fn new(insn: u32) -> Self {
        let uimm = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1f)) as i32;
        let imm = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };
        Self {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl BType {
    pub fn new(insn: u32) -> Self {
        let uimm =
            (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1e) | ((insn & 0x80) << 4)) as i32;
        let imm = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 12)
        } else {
            uimm
        };
        Self {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UType {
    pub imm: i32,
    pub rd: usize,
}

impl UType {
    pub fn new(insn: u32) -> Self {
        Self {
            imm: (insn & 0xffff_f000) as i32,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JType {
    pub imm: i32,
    pub rd: usize,
}

impl JType {
    pub fn new(insn: u32) -> Self {
        let uimm =
            ((insn & 0xff000) | ((insn & 0x100000) >> 9) | ((insn >> 20) & 0x7fe)) as i32;
        let imm = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 20)
        } else {
            uimm
        };
        Self {
            imm,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

// ── Decoded instruction enum ──────────────────────────────────────────────────

/// A decoded RISC-V instruction. Covers RV32IM and RV64IM (base + M extension).
///
/// Each variant carries the decoded format struct so callers can extract register
/// indices and immediates without re-parsing.
#[derive(Debug, Clone)]
pub enum InstructionKind {
    // ── RV32I base ────────────────────────────────────────────────────────
    // R-type ALU
    Add(RType),
    Sub(RType),
    Sll(RType),
    Slt(RType),
    Sltu(RType),
    Xor(RType),
    Srl(RType),
    Sra(RType),
    Or(RType),
    And(RType),

    // I-type ALU
    Addi(IType),
    Slti(IType),
    Sltiu(IType),
    Xori(IType),
    Ori(IType),
    Andi(IType),

    // Shift immediates (6-bit shamt for RV64 compat)
    Slli(ITypeShamt),
    Srli(ITypeShamt),
    Srai(ITypeShamt),

    // Upper immediates
    Lui(UType),
    Auipc(UType),

    // Branches
    Beq(BType),
    Bne(BType),
    Blt(BType),
    Bge(BType),
    Bltu(BType),
    Bgeu(BType),

    // Loads
    Lb(IType),
    Lh(IType),
    Lw(IType),
    Lbu(IType),
    Lhu(IType),

    // Stores
    Sb(SType),
    Sh(SType),
    Sw(SType),

    // Jumps
    Jal(JType),
    Jalr(IType),

    // Misc
    Fence(IType),

    // ── RV32M extension ───────────────────────────────────────────────────
    Mul(RType),
    Mulh(RType),
    Mulhsu(RType),
    Mulhu(RType),
    Div(RType),
    Divu(RType),
    Rem(RType),
    Remu(RType),

    // ── RV64I additions ───────────────────────────────────────────────────
    Lwu(IType),
    Ld(IType),
    Sd(SType),

    // OP-IMM-32
    Addiw(IType),
    Slliw(ITypeShamt),
    Srliw(ITypeShamt),
    Sraiw(ITypeShamt),

    // OP-32
    Addw(RType),
    Subw(RType),
    Sllw(RType),
    Srlw(RType),
    Sraw(RType),

    // ── RV64M extension ───────────────────────────────────────────────────
    Mulw(RType),
    Divw(RType),
    Divuw(RType),
    Remw(RType),
    Remuw(RType),

    // ── Unrecognised ──────────────────────────────────────────────────────
    /// The raw instruction word could not be decoded into a known instruction.
    /// Callers can inspect the raw bits to handle custom/system opcodes.
    Unknown(u32),
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/// Decode a 32-bit RISC-V instruction word into an [`InstructionKind`].
///
/// Supports RV32IM and RV64IM. System/CSR instructions are not decoded (they
/// return [`InstructionKind::Unknown`]) because the OpenVM transpiler extensions
/// handle those with custom opcode matching.
pub fn decode(insn: u32) -> InstructionKind {
    let opcode = insn & 0x7f;

    match opcode {
        OPCODE_OP => decode_op(insn),
        OPCODE_OP_IMM => decode_op_imm(insn),
        OPCODE_LUI => InstructionKind::Lui(UType::new(insn)),
        OPCODE_AUIPC => InstructionKind::Auipc(UType::new(insn)),
        OPCODE_BRANCH => decode_branch(insn),
        OPCODE_LOAD => decode_load(insn),
        OPCODE_STORE => decode_store(insn),
        OPCODE_JAL => InstructionKind::Jal(JType::new(insn)),
        OPCODE_JALR => InstructionKind::Jalr(IType::new(insn)),
        OPCODE_MISC_MEM => {
            let funct3 = (insn >> 12) & 0x7;
            if funct3 == 0b000 {
                InstructionKind::Fence(IType::new(insn))
            } else {
                InstructionKind::Unknown(insn)
            }
        }
        OPCODE_OP_32 => decode_op_32(insn),
        OPCODE_OP_IMM_32 => decode_op_imm_32(insn),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_op(insn: u32) -> InstructionKind {
    let dec = RType::new(insn);
    match (dec.funct3, dec.funct7) {
        (0b000, 0b000_0000) => InstructionKind::Add(dec),
        (0b000, 0b010_0000) => InstructionKind::Sub(dec),
        (0b001, 0b000_0000) => InstructionKind::Sll(dec),
        (0b010, 0b000_0000) => InstructionKind::Slt(dec),
        (0b011, 0b000_0000) => InstructionKind::Sltu(dec),
        (0b100, 0b000_0000) => InstructionKind::Xor(dec),
        (0b101, 0b000_0000) => InstructionKind::Srl(dec),
        (0b101, 0b010_0000) => InstructionKind::Sra(dec),
        (0b110, 0b000_0000) => InstructionKind::Or(dec),
        (0b111, 0b000_0000) => InstructionKind::And(dec),
        // M extension
        (0b000, 0b000_0001) => InstructionKind::Mul(dec),
        (0b001, 0b000_0001) => InstructionKind::Mulh(dec),
        (0b010, 0b000_0001) => InstructionKind::Mulhsu(dec),
        (0b011, 0b000_0001) => InstructionKind::Mulhu(dec),
        (0b100, 0b000_0001) => InstructionKind::Div(dec),
        (0b101, 0b000_0001) => InstructionKind::Divu(dec),
        (0b110, 0b000_0001) => InstructionKind::Rem(dec),
        (0b111, 0b000_0001) => InstructionKind::Remu(dec),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_op_imm(insn: u32) -> InstructionKind {
    let dec = IType::new(insn);
    match dec.funct3 {
        0b000 => InstructionKind::Addi(dec),
        0b010 => InstructionKind::Slti(dec),
        0b011 => InstructionKind::Sltiu(dec),
        0b100 => InstructionKind::Xori(dec),
        0b110 => InstructionKind::Ori(dec),
        0b111 => InstructionKind::Andi(dec),
        0b001 => {
            let shamt = ITypeShamt::new(insn);
            InstructionKind::Slli(shamt)
        }
        0b101 => {
            let shamt = ITypeShamt::new(insn);
            match shamt.funct6 {
                0b000000 => InstructionKind::Srli(shamt),
                0b010000 => InstructionKind::Srai(shamt),
                _ => InstructionKind::Unknown(insn),
            }
        }
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_branch(insn: u32) -> InstructionKind {
    let dec = BType::new(insn);
    match dec.funct3 {
        0b000 => InstructionKind::Beq(dec),
        0b001 => InstructionKind::Bne(dec),
        0b100 => InstructionKind::Blt(dec),
        0b101 => InstructionKind::Bge(dec),
        0b110 => InstructionKind::Bltu(dec),
        0b111 => InstructionKind::Bgeu(dec),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_load(insn: u32) -> InstructionKind {
    let dec = IType::new(insn);
    match dec.funct3 {
        0b000 => InstructionKind::Lb(dec),
        0b001 => InstructionKind::Lh(dec),
        0b010 => InstructionKind::Lw(dec),
        0b100 => InstructionKind::Lbu(dec),
        0b101 => InstructionKind::Lhu(dec),
        // RV64
        0b110 => InstructionKind::Lwu(dec),
        0b011 => InstructionKind::Ld(dec),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_store(insn: u32) -> InstructionKind {
    let dec = SType::new(insn);
    match dec.funct3 {
        0b000 => InstructionKind::Sb(dec),
        0b001 => InstructionKind::Sh(dec),
        0b010 => InstructionKind::Sw(dec),
        // RV64
        0b011 => InstructionKind::Sd(dec),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_op_32(insn: u32) -> InstructionKind {
    let dec = RType::new(insn);
    match (dec.funct3, dec.funct7) {
        (0b000, 0b000_0000) => InstructionKind::Addw(dec),
        (0b000, 0b010_0000) => InstructionKind::Subw(dec),
        (0b001, 0b000_0000) => InstructionKind::Sllw(dec),
        (0b101, 0b000_0000) => InstructionKind::Srlw(dec),
        (0b101, 0b010_0000) => InstructionKind::Sraw(dec),
        // RV64M
        (0b000, 0b000_0001) => InstructionKind::Mulw(dec),
        (0b100, 0b000_0001) => InstructionKind::Divw(dec),
        (0b101, 0b000_0001) => InstructionKind::Divuw(dec),
        (0b110, 0b000_0001) => InstructionKind::Remw(dec),
        (0b111, 0b000_0001) => InstructionKind::Remuw(dec),
        _ => InstructionKind::Unknown(insn),
    }
}

fn decode_op_imm_32(insn: u32) -> InstructionKind {
    let funct3 = (insn >> 12) & 0x7;
    match funct3 {
        0b000 => InstructionKind::Addiw(IType::new(insn)),
        0b001 => {
            let shamt = ITypeShamt::new(insn);
            if (insn >> 25) & 0x7f == 0b000_0000 {
                InstructionKind::Slliw(shamt)
            } else {
                InstructionKind::Unknown(insn)
            }
        }
        0b101 => {
            let shamt = ITypeShamt::new(insn);
            let funct7 = (insn >> 25) & 0x7f;
            match funct7 {
                0b000_0000 => InstructionKind::Srliw(shamt),
                0b010_0000 => InstructionKind::Sraiw(shamt),
                _ => InstructionKind::Unknown(insn),
            }
        }
        _ => InstructionKind::Unknown(insn),
    }
}

// ── InstructionProcessor trait ─────────────────────────────────────────────────

/// Trait for processing decoded RISC-V instructions (callback-based dispatch).
///
/// Mirrors the rrs-lib `InstructionProcessor` pattern: one method per instruction.
/// RV64-specific methods have default implementations that panic, so RV32-only
/// implementors do not need to provide them.
#[allow(unused_variables)]
pub trait InstructionProcessor {
    type InstructionResult;

    // ── RV32I base ────────────────────────────────────────────────────────
    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult;

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult;
    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult;
    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult;
    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult;

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult;
    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult;

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult;
    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult;
    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult;
    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult;
    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult;
    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult;

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult;
    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult;

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult;
    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult;
    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult;

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult;
    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult;

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult;

    // ── RV32M extension ───────────────────────────────────────────────────
    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult;
    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult;

    // ── RV64I additions (defaults panic — override for RV64) ──────────────
    fn process_lwu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        unimplemented!("LWU is RV64-only")
    }
    fn process_ld(&mut self, dec_insn: IType) -> Self::InstructionResult {
        unimplemented!("LD is RV64-only")
    }
    fn process_sd(&mut self, dec_insn: SType) -> Self::InstructionResult {
        unimplemented!("SD is RV64-only")
    }

    fn process_addw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("ADDW is RV64-only")
    }
    fn process_subw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("SUBW is RV64-only")
    }
    fn process_sllw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("SLLW is RV64-only")
    }
    fn process_srlw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("SRLW is RV64-only")
    }
    fn process_sraw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("SRAW is RV64-only")
    }

    fn process_addiw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        unimplemented!("ADDIW is RV64-only")
    }
    fn process_slliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        unimplemented!("SLLIW is RV64-only")
    }
    fn process_srliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        unimplemented!("SRLIW is RV64-only")
    }
    fn process_sraiw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        unimplemented!("SRAIW is RV64-only")
    }

    // ── RV64M additions (defaults panic — override for RV64) ──────────────
    fn process_mulw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("MULW is RV64-only")
    }
    fn process_divw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("DIVW is RV64-only")
    }
    fn process_divuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("DIVUW is RV64-only")
    }
    fn process_remw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("REMW is RV64-only")
    }
    fn process_remuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        unimplemented!("REMUW is RV64-only")
    }
}

// ── Dispatch function ─────────────────────────────────────────────────────────

fn process_opcode_op<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = RType::new(insn);
    match (dec.funct3, dec.funct7) {
        (0b000, 0b000_0000) => Some(processor.process_add(dec)),
        (0b000, 0b010_0000) => Some(processor.process_sub(dec)),
        (0b001, 0b000_0000) => Some(processor.process_sll(dec)),
        (0b010, 0b000_0000) => Some(processor.process_slt(dec)),
        (0b011, 0b000_0000) => Some(processor.process_sltu(dec)),
        (0b100, 0b000_0000) => Some(processor.process_xor(dec)),
        (0b101, 0b000_0000) => Some(processor.process_srl(dec)),
        (0b101, 0b010_0000) => Some(processor.process_sra(dec)),
        (0b110, 0b000_0000) => Some(processor.process_or(dec)),
        (0b111, 0b000_0000) => Some(processor.process_and(dec)),
        // M extension
        (0b000, 0b000_0001) => Some(processor.process_mul(dec)),
        (0b001, 0b000_0001) => Some(processor.process_mulh(dec)),
        (0b010, 0b000_0001) => Some(processor.process_mulhsu(dec)),
        (0b011, 0b000_0001) => Some(processor.process_mulhu(dec)),
        (0b100, 0b000_0001) => Some(processor.process_div(dec)),
        (0b101, 0b000_0001) => Some(processor.process_divu(dec)),
        (0b110, 0b000_0001) => Some(processor.process_rem(dec)),
        (0b111, 0b000_0001) => Some(processor.process_remu(dec)),
        _ => None,
    }
}

fn process_opcode_op_imm<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = IType::new(insn);
    match dec.funct3 {
        0b000 => Some(processor.process_addi(dec)),
        0b010 => Some(processor.process_slti(dec)),
        0b011 => Some(processor.process_sltui(dec)),
        0b100 => Some(processor.process_xori(dec)),
        0b110 => Some(processor.process_ori(dec)),
        0b111 => Some(processor.process_andi(dec)),
        0b001 => {
            let shamt = ITypeShamt::new(insn);
            if shamt.funct6 == 0b000000 {
                Some(processor.process_slli(shamt))
            } else {
                None
            }
        }
        0b101 => {
            let shamt = ITypeShamt::new(insn);
            match shamt.funct6 {
                0b000000 => Some(processor.process_srli(shamt)),
                0b010000 => Some(processor.process_srai(shamt)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn process_opcode_branch<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = BType::new(insn);
    match dec.funct3 {
        0b000 => Some(processor.process_beq(dec)),
        0b001 => Some(processor.process_bne(dec)),
        0b100 => Some(processor.process_blt(dec)),
        0b101 => Some(processor.process_bge(dec)),
        0b110 => Some(processor.process_bltu(dec)),
        0b111 => Some(processor.process_bgeu(dec)),
        _ => None,
    }
}

fn process_opcode_load<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = IType::new(insn);
    match dec.funct3 {
        0b000 => Some(processor.process_lb(dec)),
        0b001 => Some(processor.process_lh(dec)),
        0b010 => Some(processor.process_lw(dec)),
        0b100 => Some(processor.process_lbu(dec)),
        0b101 => Some(processor.process_lhu(dec)),
        // RV64
        0b011 => Some(processor.process_ld(dec)),
        0b110 => Some(processor.process_lwu(dec)),
        _ => None,
    }
}

fn process_opcode_store<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = SType::new(insn);
    match dec.funct3 {
        0b000 => Some(processor.process_sb(dec)),
        0b001 => Some(processor.process_sh(dec)),
        0b010 => Some(processor.process_sw(dec)),
        // RV64
        0b011 => Some(processor.process_sd(dec)),
        _ => None,
    }
}

fn process_opcode_op_32<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let dec = RType::new(insn);
    match (dec.funct3, dec.funct7) {
        (0b000, 0b000_0000) => Some(processor.process_addw(dec)),
        (0b000, 0b010_0000) => Some(processor.process_subw(dec)),
        (0b001, 0b000_0000) => Some(processor.process_sllw(dec)),
        (0b101, 0b000_0000) => Some(processor.process_srlw(dec)),
        (0b101, 0b010_0000) => Some(processor.process_sraw(dec)),
        // RV64M
        (0b000, 0b000_0001) => Some(processor.process_mulw(dec)),
        (0b100, 0b000_0001) => Some(processor.process_divw(dec)),
        (0b101, 0b000_0001) => Some(processor.process_divuw(dec)),
        (0b110, 0b000_0001) => Some(processor.process_remw(dec)),
        (0b111, 0b000_0001) => Some(processor.process_remuw(dec)),
        _ => None,
    }
}

fn process_opcode_op_imm_32<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let funct3 = (insn >> 12) & 0x7;
    match funct3 {
        0b000 => Some(processor.process_addiw(IType::new(insn))),
        0b001 => {
            let funct7 = (insn >> 25) & 0x7f;
            if funct7 == 0b000_0000 {
                Some(processor.process_slliw(ITypeShamt::new(insn)))
            } else {
                None
            }
        }
        0b101 => {
            let funct7 = (insn >> 25) & 0x7f;
            match funct7 {
                0b000_0000 => Some(processor.process_srliw(ITypeShamt::new(insn))),
                0b010_0000 => Some(processor.process_sraiw(ITypeShamt::new(insn))),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Decode and dispatch a 32-bit RISC-V instruction to the appropriate
/// [`InstructionProcessor`] method.
///
/// Supports RV32IM and RV64IM. Returns `None` for unrecognised encodings
/// and for system/CSR instructions (which are handled separately by
/// transpiler extensions).
pub fn process_instruction<T: InstructionProcessor>(
    processor: &mut T,
    insn: u32,
) -> Option<T::InstructionResult> {
    let opcode = insn & 0x7f;
    match opcode {
        OPCODE_OP => process_opcode_op(processor, insn),
        OPCODE_OP_IMM => process_opcode_op_imm(processor, insn),
        OPCODE_LUI => Some(processor.process_lui(UType::new(insn))),
        OPCODE_AUIPC => Some(processor.process_auipc(UType::new(insn))),
        OPCODE_BRANCH => process_opcode_branch(processor, insn),
        OPCODE_LOAD => process_opcode_load(processor, insn),
        OPCODE_STORE => process_opcode_store(processor, insn),
        OPCODE_JAL => Some(processor.process_jal(JType::new(insn))),
        OPCODE_JALR => Some(processor.process_jalr(IType::new(insn))),
        OPCODE_MISC_MEM => {
            let funct3 = (insn >> 12) & 0x7;
            if funct3 == 0b000 {
                Some(processor.process_fence(IType::new(insn)))
            } else {
                None
            }
        }
        // RV64
        OPCODE_OP_32 => process_opcode_op_32(processor, insn),
        OPCODE_OP_IMM_32 => process_opcode_op_imm_32(processor, insn),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Format struct tests (ported from rrs-lib) ─────────────────────────

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
        );
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
        // ori x13, x7, -2048
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
        let s = ITypeShamt::new(0x00d29613);
        assert_eq!(s.shamt, 13);
        assert_eq!(s.rs1, 5);
        assert_eq!(s.funct3, 0b001);
        assert_eq!(s.rd, 12);

        // srli x30, x19, 31
        let s = ITypeShamt::new(0x01f9df13);
        assert_eq!(s.shamt, 31);
        assert_eq!(s.rs1, 19);
        assert_eq!(s.funct3, 0b101);
        assert_eq!(s.rd, 30);
    }

    #[test]
    fn test_itype_shamt_6bit() {
        // slli x1, x2, 32 (RV64 only — bit 25 set)
        // Encoding: funct6=0b000000, shamt=0b100000, rs1=x2, funct3=001, rd=x1
        let insn = 0b000000_100000_00010_001_00001_0010011u32;
        let s = ITypeShamt::new(insn);
        assert_eq!(s.shamt, 32);
        assert_eq!(s.funct6, 0b000000);
        assert_eq!(s.rs1, 2);
        assert_eq!(s.rd, 1);
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

    // ── Decode tests ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_add() {
        // add x1, x2, x3
        let insn = 0x003100b3;
        match decode(insn) {
            InstructionKind::Add(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_sub() {
        // sub x1, x2, x3
        let insn = 0x403100b3;
        match decode(insn) {
            InstructionKind::Sub(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Sub, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_addi() {
        // addi x1, x2, 42
        let insn = 0x02a10093;
        match decode(insn) {
            InstructionKind::Addi(i) => {
                assert_eq!(i.rd, 1);
                assert_eq!(i.rs1, 2);
                assert_eq!(i.imm, 42);
            }
            other => panic!("expected Addi, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_beq() {
        // beq x10, x14, .-4096
        let insn = 0x80e50063;
        match decode(insn) {
            InstructionKind::Beq(b) => {
                assert_eq!(b.rs1, 10);
                assert_eq!(b.rs2, 14);
                assert_eq!(b.imm, -4096);
            }
            other => panic!("expected Beq, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_lw() {
        // lw x1, 4(x2)
        let insn = 0x00412083;
        match decode(insn) {
            InstructionKind::Lw(i) => {
                assert_eq!(i.rd, 1);
                assert_eq!(i.rs1, 2);
                assert_eq!(i.imm, 4);
            }
            other => panic!("expected Lw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_sw() {
        // sw x8, 1(x23)
        let insn = 0x008ba0a3;
        match decode(insn) {
            InstructionKind::Sw(s) => {
                assert_eq!(s.rs2, 8);
                assert_eq!(s.rs1, 23);
                assert_eq!(s.imm, 1);
            }
            other => panic!("expected Sw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_jal() {
        // jal x26, .+46
        let insn = 0x02e00d6f;
        match decode(insn) {
            InstructionKind::Jal(j) => {
                assert_eq!(j.rd, 26);
                assert_eq!(j.imm, 46);
            }
            other => panic!("expected Jal, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_lui() {
        // lui x17, 0x123ab
        let insn = 0x123ab8b7;
        match decode(insn) {
            InstructionKind::Lui(u) => {
                assert_eq!(u.rd, 17);
                assert_eq!(u.imm, 0x123ab000);
            }
            other => panic!("expected Lui, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_mul() {
        // mul x1, x2, x3
        let insn = 0x023100b3;
        match decode(insn) {
            InstructionKind::Mul(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_slli_rv64() {
        // slli x1, x2, 32 (6-bit shamt)
        let insn = 0b000000_100000_00010_001_00001_0010011u32;
        match decode(insn) {
            InstructionKind::Slli(s) => {
                assert_eq!(s.rd, 1);
                assert_eq!(s.rs1, 2);
                assert_eq!(s.shamt, 32);
                assert_eq!(s.funct6, 0b000000);
            }
            other => panic!("expected Slli, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_ld() {
        // ld x1, 8(x3) — funct3=011, opcode=LOAD
        let insn = 0x0081b083;
        match decode(insn) {
            InstructionKind::Ld(i) => {
                assert_eq!(i.rd, 1);
                assert_eq!(i.rs1, 3);
                assert_eq!(i.imm, 8);
            }
            other => panic!("expected Ld, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_sd() {
        // sd x3, 16(x2) — funct3=011, opcode=STORE
        let insn = 0x00313823;
        match decode(insn) {
            InstructionKind::Sd(s) => {
                assert_eq!(s.rs2, 3);
                assert_eq!(s.rs1, 2);
                assert_eq!(s.imm, 16);
            }
            other => panic!("expected Sd, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_addw() {
        // addw x1, x2, x3 — opcode=OP-32, funct3=000, funct7=0000000
        let insn = 0x003100bb;
        match decode(insn) {
            InstructionKind::Addw(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Addw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_subw() {
        // subw x1, x2, x3 — opcode=OP-32, funct3=000, funct7=0100000
        let insn = 0x403100bb;
        match decode(insn) {
            InstructionKind::Subw(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Subw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_addiw() {
        // addiw x1, x2, 42 — opcode=OP-IMM-32, funct3=000
        let insn = 0x02a1009b;
        match decode(insn) {
            InstructionKind::Addiw(i) => {
                assert_eq!(i.rd, 1);
                assert_eq!(i.rs1, 2);
                assert_eq!(i.imm, 42);
            }
            other => panic!("expected Addiw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_unknown_system() {
        // ecall — opcode=SYSTEM, should be Unknown
        let insn = 0x00000073;
        assert!(matches!(decode(insn), InstructionKind::Unknown(_)));
    }

    #[test]
    fn test_decode_fence() {
        // fence instruction
        let insn = 0x0ff0000f;
        assert!(matches!(decode(insn), InstructionKind::Fence(_)));
    }

    #[test]
    fn test_decode_mulw() {
        // mulw x1, x2, x3 — opcode=OP-32, funct3=000, funct7=0000001
        let insn = 0x023100bb;
        match decode(insn) {
            InstructionKind::Mulw(r) => {
                assert_eq!(r.rd, 1);
                assert_eq!(r.rs1, 2);
                assert_eq!(r.rs2, 3);
            }
            other => panic!("expected Mulw, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_lwu() {
        // lwu x1, 4(x2) — funct3=110, opcode=LOAD
        let insn = 0x00416083;
        match decode(insn) {
            InstructionKind::Lwu(i) => {
                assert_eq!(i.rd, 1);
                assert_eq!(i.rs1, 2);
                assert_eq!(i.imm, 4);
            }
            other => panic!("expected Lwu, got {:?}", other),
        }
    }
}
