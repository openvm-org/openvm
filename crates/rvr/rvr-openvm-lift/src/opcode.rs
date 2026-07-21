//! OpenVM opcode -> rvr-openvm-ir mapping for RISC-V instructions.
//!
//! Supports the basic OpenVM ISA:
//! - RISC-V base instructions
//! - System instructions: TERMINATE, PHANTOM, PUBLISH
//! - System phantom sub-instructions: Nop, DebugPanic, CtStart, CtEnd
//! - RV64 load/store instructions lift memory address space `e = 2`.
//! - RV64 stores to public values address space `e = 3`, including REVEAL, are handled by
//!   extensions.

use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SysPhantom, SystemOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode,
    Rv64LoadStoreOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
};
use rvr_openvm_ir::{
    AluOp, BranchCond, Instr, InstrAt, LiftedInstr, MemWidth, MulDivOp, Terminator,
};

use crate::{
    helpers::{decode_imm_cg, decode_reg, sext32},
    ExtensionRegistry, RvrInstruction,
};

const U24_MASK: u32 = (1 << 24) - 1;

/// Lift a single OpenVM instruction to the new IR types.
///
/// Returns `None` for unrecognized opcodes. If a registered extension handles
/// the opcode, its `try_lift` is called.
pub fn lift_instruction(
    insn: &RvrInstruction,
    pc: u64,
    extensions: &ExtensionRegistry,
) -> Option<LiftedInstr> {
    let opcode = insn.opcode.as_usize();

    // System opcodes
    if opcode == SystemOpcode::TERMINATE.global_opcode_usize() {
        let exit_code = insn.c;
        return Some(LiftedInstr::Term {
            pc,
            terminator: Terminator::Exit { code: exit_code },
            source_loc: None,
        });
    }
    if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
        let discriminant = (insn.c & 0xffff) as u16;
        if let Some(sys) = SysPhantom::from_repr(discriminant) {
            return Some(lift_phantom(sys, pc));
        }
        if let Some(lifted) = extensions.try_lift(insn, pc) {
            return Some(lifted);
        }
        return None;
    }

    if opcode == BaseAluOpcode::ADD.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Add);
    }
    if opcode == BaseAluOpcode::SUB.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Sub);
    }

    if opcode == BaseAluOpcode::XOR.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Xor);
    }
    if opcode == BaseAluOpcode::OR.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Or);
    }
    if opcode == BaseAluOpcode::AND.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::And);
    }
    if opcode == BaseAluWOpcode::ADDW.global_opcode_usize() {
        return lift_alu_w_reg(insn, pc, AluOp::Add);
    }
    if opcode == BaseAluWOpcode::SUBW.global_opcode_usize() {
        return lift_alu_w_reg(insn, pc, AluOp::Sub);
    }
    if opcode == BaseAluWImmOpcode::ADDIW.global_opcode_usize() {
        return lift_alu_w_imm(insn, pc, AluOp::Add);
    }

    if opcode == BaseAluImmOpcode::ADDI.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::Add);
    }

    if opcode == BaseAluImmOpcode::XORI.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::Xor);
    }
    if opcode == BaseAluImmOpcode::ORI.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::Or);
    }
    if opcode == BaseAluImmOpcode::ANDI.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::And);
    }

    if opcode == LessThanImmOpcode::SLTI.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::Slt);
    }
    if opcode == LessThanImmOpcode::SLTIU.global_opcode_usize() {
        return lift_alu_imm(insn, pc, AluOp::Sltu);
    }

    if opcode == ShiftImmOpcode::SLLI.global_opcode_usize() {
        return lift_shift_imm(insn, pc, AluOp::Sll);
    }
    if opcode == ShiftImmOpcode::SRLI.global_opcode_usize() {
        return lift_shift_imm(insn, pc, AluOp::Srl);
    }
    if opcode == ShiftImmOpcode::SRAI.global_opcode_usize() {
        return lift_shift_imm(insn, pc, AluOp::Sra);
    }

    if opcode == ShiftOpcode::SLL.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Sll);
    }
    if opcode == ShiftOpcode::SRL.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Srl);
    }
    if opcode == ShiftOpcode::SRA.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Sra);
    }
    if opcode == ShiftWOpcode::SLLW.global_opcode_usize() {
        return lift_alu_w_reg(insn, pc, AluOp::Sll);
    }
    if opcode == ShiftWOpcode::SRLW.global_opcode_usize() {
        return lift_alu_w_reg(insn, pc, AluOp::Srl);
    }
    if opcode == ShiftWOpcode::SRAW.global_opcode_usize() {
        return lift_alu_w_reg(insn, pc, AluOp::Sra);
    }
    if opcode == ShiftWImmOpcode::SLLIW.global_opcode_usize() {
        return lift_shift_w_imm(insn, pc, AluOp::Sll);
    }
    if opcode == ShiftWImmOpcode::SRLIW.global_opcode_usize() {
        return lift_shift_w_imm(insn, pc, AluOp::Srl);
    }
    if opcode == ShiftWImmOpcode::SRAIW.global_opcode_usize() {
        return lift_shift_w_imm(insn, pc, AluOp::Sra);
    }

    if opcode == LessThanOpcode::SLT.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Slt);
    }
    if opcode == LessThanOpcode::SLTU.global_opcode_usize() {
        return lift_alu_reg(insn, pc, AluOp::Sltu);
    }

    // LoadStore: LOADD=0x210, LOADBU=0x211, LOADHU=0x212, LOADWU=0x213,
    //            STORED=0x214, STOREW=0x215, STOREH=0x216, STOREB=0x217,
    //            LOADB=0x218, LOADH=0x219, LOADW=0x21a
    if opcode == Rv64LoadStoreOpcode::LOADD.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Double, false);
    }
    if opcode == Rv64LoadStoreOpcode::LOADBU.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Byte, false);
    }
    if opcode == Rv64LoadStoreOpcode::LOADHU.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Half, false);
    }
    if opcode == Rv64LoadStoreOpcode::LOADWU.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Word, false);
    }
    if opcode == Rv64LoadStoreOpcode::LOADB.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Byte, true);
    }
    if opcode == Rv64LoadStoreOpcode::LOADH.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Half, true);
    }
    if opcode == Rv64LoadStoreOpcode::LOADW.global_opcode_usize() {
        return lift_load(insn, pc, MemWidth::Word, true);
    }
    if opcode == Rv64LoadStoreOpcode::STORED.global_opcode_usize() {
        if let Some(lifted) = lift_store(insn, pc, MemWidth::Double) {
            return Some(lifted);
        }
        return lift_public_values_store(insn, pc, extensions);
    }
    if opcode == Rv64LoadStoreOpcode::STOREW.global_opcode_usize() {
        if let Some(lifted) = lift_store(insn, pc, MemWidth::Word) {
            return Some(lifted);
        }
        return lift_public_values_store(insn, pc, extensions);
    }
    if opcode == Rv64LoadStoreOpcode::STOREH.global_opcode_usize() {
        if let Some(lifted) = lift_store(insn, pc, MemWidth::Half) {
            return Some(lifted);
        }
        return lift_public_values_store(insn, pc, extensions);
    }
    if opcode == Rv64LoadStoreOpcode::STOREB.global_opcode_usize() {
        if let Some(lifted) = lift_store(insn, pc, MemWidth::Byte) {
            return Some(lifted);
        }
        return lift_public_values_store(insn, pc, extensions);
    }

    // BranchEqual: BEQ=0x220, BNE=0x221
    if opcode == BranchEqualOpcode::BEQ.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Eq));
    }
    if opcode == BranchEqualOpcode::BNE.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Ne));
    }

    // BranchLT: BLT=0x225, BLTU=0x226, BGE=0x227, BGEU=0x228
    if opcode == BranchLessThanOpcode::BLT.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Lt));
    }
    if opcode == BranchLessThanOpcode::BLTU.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Ltu));
    }
    if opcode == BranchLessThanOpcode::BGE.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Ge));
    }
    if opcode == BranchLessThanOpcode::BGEU.global_opcode_usize() {
        return Some(lift_branch(insn, pc, BranchCond::Geu));
    }

    // JAL=0x230
    if opcode == Rv64JalLuiOpcode::JAL.global_opcode_usize() {
        return Some(lift_jal(insn, pc));
    }
    // LUI=0x231
    if opcode == Rv64JalLuiOpcode::LUI.global_opcode_usize() {
        return Some(lift_lui(insn, pc));
    }
    // JALR=0x235
    if opcode == Rv64JalrOpcode::JALR.global_opcode_usize() {
        return Some(lift_jalr(insn, pc));
    }
    // AUIPC=0x240
    if opcode == Rv64AuipcOpcode::AUIPC.global_opcode_usize() {
        return Some(lift_auipc(insn, pc));
    }

    // MUL=0x250
    if opcode == MulOpcode::MUL.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Mul));
    }
    // MULW=0x280
    if opcode == MulWOpcode::MULW.global_opcode_usize() {
        return Some(lift_muldiv_w(insn, pc, MulDivOp::Mul));
    }
    // MULH=0x251, MULHSU=0x252, MULHU=0x253
    if opcode == MulHOpcode::MULH.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Mulh));
    }
    if opcode == MulHOpcode::MULHSU.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Mulhsu));
    }
    if opcode == MulHOpcode::MULHU.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Mulhu));
    }
    // DIV=0x254, DIVU=0x255, REM=0x256, REMU=0x257
    // DIVW=0x284, DIVUW=0x285, REMW=0x286, REMUW=0x287
    if opcode == DivRemOpcode::DIV.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Div));
    }
    if opcode == DivRemOpcode::DIVU.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Divu));
    }
    if opcode == DivRemOpcode::REM.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Rem));
    }
    if opcode == DivRemOpcode::REMU.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Remu));
    }
    if opcode == DivRemWOpcode::DIVW.global_opcode_usize() {
        return Some(lift_muldiv_w(insn, pc, MulDivOp::Div));
    }
    if opcode == DivRemWOpcode::DIVUW.global_opcode_usize() {
        return Some(lift_muldiv_w(insn, pc, MulDivOp::Divu));
    }
    if opcode == DivRemWOpcode::REMW.global_opcode_usize() {
        return Some(lift_muldiv_w(insn, pc, MulDivOp::Rem));
    }
    if opcode == DivRemWOpcode::REMUW.global_opcode_usize() {
        return Some(lift_muldiv_w(insn, pc, MulDivOp::Remu));
    }

    // Fall through to registered extensions.
    extensions.try_lift(insn, pc)
}

// ============= Helpers =============

/// Sign-extend a 12-bit immediate stored in the low 24 bits.
fn sign_extend_12(val: u32) -> i32 {
    let val12 = val & 0xfff;
    if val12 & 0x800 != 0 {
        (val12 | 0xfffff000) as i32
    } else {
        val12 as i32
    }
}

// ============= Instruction Lifters =============

/// Helper to create a body instruction.
pub fn body(pc: u64, instr: Instr) -> LiftedInstr {
    LiftedInstr::Body(InstrAt {
        pc,
        instr,
        source_loc: None,
    })
}

/// Helper to create a terminator instruction.
pub fn term(pc: u64, terminator: Terminator) -> LiftedInstr {
    LiftedInstr::Term {
        pc,
        terminator,
        source_loc: None,
    }
}

fn lift_alu_reg(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_REGISTER_AS {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let rs2 = decode_reg(insn.c);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(pc, Instr::AluReg { op, rd, rs1, rs2 }))
}

fn lift_alu_imm(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
        return None;
    }

    let raw_imm = insn.c;
    let imm = sign_extend_12(raw_imm);
    if raw_imm != (imm as u32 & U24_MASK) {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(pc, Instr::AluImm { op, rd, rs1, imm }))
}

fn lift_shift_imm(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
        return None;
    }

    let shamt = insn.c;
    if shamt >= 64 {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(
        pc,
        Instr::ShiftImm {
            op,
            rd,
            rs1,
            shamt: shamt as u8,
        },
    ))
}

/// Lift MUL/DIV/REM R-type instruction.
fn lift_muldiv(insn: &RvrInstruction, pc: u64, op: MulDivOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let rs2 = decode_reg(insn.c);
    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::MulDiv { op, rd, rs1, rs2 })
}

#[inline]
fn is_memory_instruction(insn: &RvrInstruction) -> bool {
    insn.d == RV64_REGISTER_AS && insn.e == RV64_MEMORY_AS
}

#[inline]
fn is_public_values_instruction(insn: &RvrInstruction) -> bool {
    insn.d == RV64_REGISTER_AS && insn.e == PUBLIC_VALUES_AS
}

fn lift_public_values_store(
    insn: &RvrInstruction,
    pc: u64,
    extensions: &ExtensionRegistry,
) -> Option<LiftedInstr> {
    if is_public_values_instruction(insn) {
        extensions.try_lift(insn, pc)
    } else {
        None
    }
}

/// Lift load instruction.
/// OpenVM encoding: rd=a/8, rs1=b/8, imm low16=c, sign=g!=0
fn lift_load(insn: &RvrInstruction, pc: u64, width: MemWidth, signed: bool) -> Option<LiftedInstr> {
    if !is_memory_instruction(insn) {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let offset = decode_imm_cg(insn) as i16;

    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(
        pc,
        Instr::Load {
            width,
            signed,
            rd,
            rs1,
            offset,
        },
    ))
}

/// Lift store instruction.
/// OpenVM encoding: rs2=a/8, rs1=b/8, imm low16=c, sign=g!=0
fn lift_store(insn: &RvrInstruction, pc: u64, width: MemWidth) -> Option<LiftedInstr> {
    if !is_memory_instruction(insn) {
        return None;
    }

    let rs2 = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let offset = decode_imm_cg(insn) as i16;

    Some(body(
        pc,
        Instr::Store {
            width,
            rs1,
            rs2,
            offset,
        },
    ))
}

/// Lift branch instruction.
/// OpenVM encoding: rs1=a/8, rs2=b/8, offset=c as signed (BabyBear modular)
fn lift_branch(insn: &RvrInstruction, pc: u64, cond: BranchCond) -> LiftedInstr {
    let rs1 = decode_reg(insn.a);
    let rs2 = decode_reg(insn.b);
    let offset = insn.signed_c();
    let target = (pc as i64 + offset as i64) as u64;

    term(
        pc,
        Terminator::Branch {
            cond,
            rs1,
            rs2,
            target,
        },
    )
}

/// Lift JAL instruction.
/// OpenVM encoding: rd=a/8, offset=c as signed (BabyBear modular)
fn lift_jal(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let offset = insn.signed_c();
    let target = (pc as i64 + offset as i64) as u64;

    let link_rd = if rd != 0 { Some(rd) } else { None };
    term(pc, Terminator::Jump { link_rd, target })
}

/// Lift JALR instruction.
/// OpenVM encoding: rd=a/8, rs1=b/8, imm low16=c, sign=g!=0
fn lift_jalr(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let imm = decode_imm_cg(insn) as i32;

    let link_rd = if rd != 0 { Some(rd) } else { None };
    term(
        pc,
        Terminator::JumpDyn {
            link_rd,
            rs1,
            imm,
            resolved: vec![],
        },
    )
}

/// Lift LUI instruction.
/// OpenVM encoding: rd=a/8, upper20=c << 12
fn lift_lui(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let upper = insn.c;
    let value = upper << 12;

    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::Lui { rd, value })
}

/// Lift AUIPC instruction.
/// OpenVM encoding: rd=a/8, upper20 = c << 8 (from rrs.rs: c = (imm & 0xfffff000) >> 8)
fn lift_auipc(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let shifted = insn.c;
    let upper = shifted << 8; // reconstruct the upper 20 bits with low 12 zeros
    let value = pc.wrapping_add(sext32(upper));

    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::Auipc { rd, value })
}

/// Lift a register-register W-suffix ALU instruction.
/// Result is the low 32 bits of the operation, sign-extended to 64 bits.
fn lift_alu_w_reg(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_REGISTER_AS {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let rs2 = decode_reg(insn.c);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(pc, Instr::AluWReg { op, rd, rs1, rs2 }))
}

/// Lift an immediate W-suffix ALU instruction.
fn lift_alu_w_imm(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
        return None;
    }

    let raw_imm = insn.c;
    let imm = sign_extend_12(raw_imm);
    if raw_imm != (imm as u32 & U24_MASK) {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(pc, Instr::AluWImm { op, rd, rs1, imm }))
}

/// Lift an immediate W-suffix shift instruction.
/// Shamt is 5-bit (W shifts operate on 32-bit values regardless of register width).
fn lift_shift_w_imm(insn: &RvrInstruction, pc: u64, op: AluOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
        return None;
    }

    let shamt = insn.c;
    if shamt >= 32 {
        return None;
    }

    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    if rd == 0 {
        return Some(body(pc, Instr::Nop));
    }
    Some(body(
        pc,
        Instr::ShiftWImm {
            op,
            rd,
            rs1,
            shamt: shamt as u8,
        },
    ))
}

/// Lift W-suffix MUL/DIV/REM R-type instruction.
fn lift_muldiv_w(insn: &RvrInstruction, pc: u64, op: MulDivOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let rs2 = decode_reg(insn.c);
    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::MulDivW { op, rd, rs1, rs2 })
}

// ============= System / IO Instructions =============

/// Lift a system PHANTOM sub-instruction.
fn lift_phantom(sys: SysPhantom, pc: u64) -> LiftedInstr {
    match sys {
        // Nop, CtStart, CtEnd — no-ops for execution
        SysPhantom::Nop | SysPhantom::CtStart | SysPhantom::CtEnd => body(pc, Instr::Nop),

        // DebugPanic — trap on host
        SysPhantom::DebugPanic => term(
            pc,
            Terminator::Trap {
                message: "PHANTOM DebugPanic".to_string(),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, VmOpcode, DEFERRAL_AS, PUBLIC_VALUES_AS,
    };
    use openvm_riscv_transpiler::{
        BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
        LessThanImmOpcode, LessThanOpcode, Rv64AuipcOpcode, Rv64JalLuiOpcode,
        Rv64LoadStoreOpcode::{
            LOADB, LOADBU, LOADD, LOADH, LOADHU, LOADW, LOADWU, STOREB, STORED, STOREH, STOREW,
        },
        ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
    };
    use p3_baby_bear::BabyBear;

    use super::{lift_instruction, AluOp, Instr, InstrAt, LiftedInstr, Terminator};
    use crate::{ExtensionRegistry, RvrInstruction};

    fn lift_babybear(
        insn: &Instruction<BabyBear>,
        pc: u64,
        extensions: &ExtensionRegistry,
    ) -> Option<LiftedInstr> {
        lift_instruction(&RvrInstruction::from_field(insn), pc, extensions)
    }

    fn base_alu_reg_instruction(opcode: BaseAluOpcode, e: u32) -> Instruction<BabyBear> {
        Instruction::from_usize(
            opcode.global_opcode(),
            [
                RV64_REGISTER_NUM_LIMBS,
                2 * RV64_REGISTER_NUM_LIMBS,
                3 * RV64_REGISTER_NUM_LIMBS,
                RV64_REGISTER_AS as usize,
                e as usize,
            ],
        )
    }

    fn alu_instruction(opcode: VmOpcode, c: usize, d: u32, e: u32) -> Instruction<BabyBear> {
        Instruction::from_usize(
            opcode,
            [
                RV64_REGISTER_NUM_LIMBS,
                2 * RV64_REGISTER_NUM_LIMBS,
                c,
                d as usize,
                e as usize,
            ],
        )
    }

    #[test]
    fn add_sub_require_register_operands() {
        let extensions = ExtensionRegistry::new();

        for (opcode, expected_op) in [
            (BaseAluOpcode::ADD, AluOp::Add),
            (BaseAluOpcode::SUB, AluOp::Sub),
        ] {
            let valid = base_alu_reg_instruction(opcode, RV64_REGISTER_AS);
            match lift_babybear(&valid, 0x100, &extensions) {
                Some(LiftedInstr::Body(InstrAt {
                    instr: Instr::AluReg { op, .. },
                    ..
                })) => assert_eq!(op, expected_op),
                other => panic!("unexpected lift: {other:?}"),
            }

            let immediate = base_alu_reg_instruction(opcode, RV64_IMM_AS);
            assert!(lift_babybear(&immediate, 0x100, &extensions).is_none());
        }
    }

    #[test]
    fn addi_requires_canonical_immediate_encoding() {
        let extensions = ExtensionRegistry::new();
        let valid = alu_instruction(
            BaseAluImmOpcode::ADDI.global_opcode(),
            0xff_ffff,
            RV64_REGISTER_AS,
            RV64_IMM_AS,
        );

        match lift_babybear(&valid, 0x100, &extensions) {
            Some(LiftedInstr::Body(InstrAt {
                instr:
                    Instr::AluImm {
                        op: AluOp::Add,
                        rd: 1,
                        rs1: 2,
                        imm: -1,
                    },
                ..
            })) => {}
            other => panic!("unexpected lift: {other:?}"),
        }

        let register_operand = alu_instruction(
            BaseAluImmOpcode::ADDI.global_opcode(),
            3 * RV64_REGISTER_NUM_LIMBS,
            RV64_REGISTER_AS,
            RV64_REGISTER_AS,
        );
        assert!(lift_babybear(&register_operand, 0x100, &extensions).is_none());

        let noncanonical_immediate = alu_instruction(
            BaseAluImmOpcode::ADDI.global_opcode(),
            0xffff,
            RV64_REGISTER_AS,
            RV64_IMM_AS,
        );
        assert!(lift_babybear(&noncanonical_immediate, 0x100, &extensions).is_none());
    }

    #[test]
    fn immediate_alu_families_require_canonical_i12_encoding() {
        let extensions = ExtensionRegistry::new();
        let opcodes = [
            (BaseAluImmOpcode::XORI.global_opcode(), AluOp::Xor),
            (BaseAluImmOpcode::ORI.global_opcode(), AluOp::Or),
            (BaseAluImmOpcode::ANDI.global_opcode(), AluOp::And),
            (LessThanImmOpcode::SLTI.global_opcode(), AluOp::Slt),
            (LessThanImmOpcode::SLTIU.global_opcode(), AluOp::Sltu),
        ];

        for (opcode, expected_op) in opcodes {
            for (c, expected_imm) in [(0, 0), (0x7ff, 0x7ff), (0xff_f800, -0x800), (0xff_ffff, -1)]
            {
                let instruction = alu_instruction(opcode, c, RV64_REGISTER_AS, RV64_IMM_AS);
                match lift_babybear(&instruction, 0x100, &extensions) {
                    Some(LiftedInstr::Body(InstrAt {
                        instr: Instr::AluImm { op, rd, rs1, imm },
                        ..
                    })) => assert_eq!((op, rd, rs1, imm), (expected_op, 1, 2, expected_imm)),
                    other => panic!("unexpected lift: {other:?}"),
                }
            }

            for instruction in [
                alu_instruction(opcode, 0, RV64_IMM_AS, RV64_IMM_AS),
                alu_instruction(opcode, 0, RV64_REGISTER_AS, RV64_REGISTER_AS),
                alu_instruction(opcode, 0x800, RV64_REGISTER_AS, RV64_IMM_AS),
                alu_instruction(opcode, 0xffff, RV64_REGISTER_AS, RV64_IMM_AS),
            ] {
                assert!(lift_babybear(&instruction, 0x100, &extensions).is_none());
            }
        }
    }

    #[test]
    fn shift_immediates_require_six_bit_amount() {
        let extensions = ExtensionRegistry::new();

        for (opcode, expected_op) in [
            (ShiftImmOpcode::SLLI.global_opcode(), AluOp::Sll),
            (ShiftImmOpcode::SRLI.global_opcode(), AluOp::Srl),
            (ShiftImmOpcode::SRAI.global_opcode(), AluOp::Sra),
        ] {
            for shamt in [0, 63] {
                let instruction = alu_instruction(opcode, shamt, RV64_REGISTER_AS, RV64_IMM_AS);
                match lift_babybear(&instruction, 0x100, &extensions) {
                    Some(LiftedInstr::Body(InstrAt {
                        instr:
                            Instr::ShiftImm {
                                op,
                                rd,
                                rs1,
                                shamt: lifted_shamt,
                            },
                        ..
                    })) => assert_eq!(
                        (op, rd, rs1, lifted_shamt),
                        (expected_op, 1, 2, shamt as u8)
                    ),
                    other => panic!("unexpected lift: {other:?}"),
                }
            }

            for instruction in [
                alu_instruction(opcode, 0, RV64_IMM_AS, RV64_IMM_AS),
                alu_instruction(opcode, 0, RV64_REGISTER_AS, RV64_REGISTER_AS),
                alu_instruction(opcode, 64, RV64_REGISTER_AS, RV64_IMM_AS),
            ] {
                assert!(lift_babybear(&instruction, 0x100, &extensions).is_none());
            }
        }
    }

    #[test]
    fn word_alu_opcodes_separate_register_and_immediate_forms() {
        let extensions = ExtensionRegistry::new();

        for (opcode, expected_op) in [
            (BaseAluWOpcode::ADDW.global_opcode(), AluOp::Add),
            (BaseAluWOpcode::SUBW.global_opcode(), AluOp::Sub),
        ] {
            let register = alu_instruction(
                opcode,
                3 * RV64_REGISTER_NUM_LIMBS,
                RV64_REGISTER_AS,
                RV64_REGISTER_AS,
            );
            match lift_babybear(&register, 0x100, &extensions) {
                Some(LiftedInstr::Body(InstrAt {
                    instr: Instr::AluWReg { op, rd, rs1, rs2 },
                    ..
                })) => assert_eq!((op, rd, rs1, rs2), (expected_op, 1, 2, 3)),
                other => panic!("unexpected lift: {other:?}"),
            }

            let immediate = alu_instruction(opcode, 0, RV64_REGISTER_AS, RV64_IMM_AS);
            assert!(lift_babybear(&immediate, 0x100, &extensions).is_none());
        }

        for (c, expected_imm) in [(0, 0), (0x7ff, 0x7ff), (0xff_f800, -0x800), (0xff_ffff, -1)] {
            let instruction = alu_instruction(
                BaseAluWImmOpcode::ADDIW.global_opcode(),
                c,
                RV64_REGISTER_AS,
                RV64_IMM_AS,
            );
            match lift_babybear(&instruction, 0x100, &extensions) {
                Some(LiftedInstr::Body(InstrAt {
                    instr: Instr::AluWImm { op, rd, rs1, imm },
                    ..
                })) => assert_eq!((op, rd, rs1, imm), (AluOp::Add, 1, 2, expected_imm)),
                other => panic!("unexpected lift: {other:?}"),
            }
        }

        for instruction in [
            alu_instruction(
                BaseAluWImmOpcode::ADDIW.global_opcode(),
                0,
                RV64_IMM_AS,
                RV64_IMM_AS,
            ),
            alu_instruction(
                BaseAluWImmOpcode::ADDIW.global_opcode(),
                0,
                RV64_REGISTER_AS,
                RV64_REGISTER_AS,
            ),
            alu_instruction(
                BaseAluWImmOpcode::ADDIW.global_opcode(),
                0xffff,
                RV64_REGISTER_AS,
                RV64_IMM_AS,
            ),
        ] {
            assert!(lift_babybear(&instruction, 0x100, &extensions).is_none());
        }
    }

    #[test]
    fn word_shift_opcodes_separate_register_and_immediate_forms() {
        let extensions = ExtensionRegistry::new();

        for (opcode, expected_op) in [
            (ShiftWOpcode::SLLW.global_opcode(), AluOp::Sll),
            (ShiftWOpcode::SRLW.global_opcode(), AluOp::Srl),
            (ShiftWOpcode::SRAW.global_opcode(), AluOp::Sra),
        ] {
            let register = alu_instruction(
                opcode,
                3 * RV64_REGISTER_NUM_LIMBS,
                RV64_REGISTER_AS,
                RV64_REGISTER_AS,
            );
            match lift_babybear(&register, 0x100, &extensions) {
                Some(LiftedInstr::Body(InstrAt {
                    instr: Instr::AluWReg { op, rd, rs1, rs2 },
                    ..
                })) => assert_eq!((op, rd, rs1, rs2), (expected_op, 1, 2, 3)),
                other => panic!("unexpected lift: {other:?}"),
            }

            let immediate = alu_instruction(opcode, 0, RV64_REGISTER_AS, RV64_IMM_AS);
            assert!(lift_babybear(&immediate, 0x100, &extensions).is_none());
        }

        for (opcode, expected_op) in [
            (ShiftWImmOpcode::SLLIW.global_opcode(), AluOp::Sll),
            (ShiftWImmOpcode::SRLIW.global_opcode(), AluOp::Srl),
            (ShiftWImmOpcode::SRAIW.global_opcode(), AluOp::Sra),
        ] {
            for shamt in [0, 31] {
                let instruction = alu_instruction(opcode, shamt, RV64_REGISTER_AS, RV64_IMM_AS);
                match lift_babybear(&instruction, 0x100, &extensions) {
                    Some(LiftedInstr::Body(InstrAt {
                        instr:
                            Instr::ShiftWImm {
                                op,
                                rd,
                                rs1,
                                shamt: lifted_shamt,
                            },
                        ..
                    })) => assert_eq!(
                        (op, rd, rs1, lifted_shamt),
                        (expected_op, 1, 2, shamt as u8)
                    ),
                    other => panic!("unexpected lift: {other:?}"),
                }
            }

            for instruction in [
                alu_instruction(opcode, 0, RV64_IMM_AS, RV64_IMM_AS),
                alu_instruction(opcode, 0, RV64_REGISTER_AS, RV64_REGISTER_AS),
                alu_instruction(opcode, 32, RV64_REGISTER_AS, RV64_IMM_AS),
            ] {
                assert!(lift_babybear(&instruction, 0x100, &extensions).is_none());
            }
        }
    }

    #[test]
    fn register_alu_families_reject_immediate_encoding() {
        let extensions = ExtensionRegistry::new();

        for (opcode, expected_op) in [
            (BaseAluOpcode::XOR.global_opcode(), AluOp::Xor),
            (BaseAluOpcode::OR.global_opcode(), AluOp::Or),
            (BaseAluOpcode::AND.global_opcode(), AluOp::And),
            (LessThanOpcode::SLT.global_opcode(), AluOp::Slt),
            (LessThanOpcode::SLTU.global_opcode(), AluOp::Sltu),
            (ShiftOpcode::SLL.global_opcode(), AluOp::Sll),
            (ShiftOpcode::SRL.global_opcode(), AluOp::Srl),
            (ShiftOpcode::SRA.global_opcode(), AluOp::Sra),
        ] {
            let register = alu_instruction(
                opcode,
                3 * RV64_REGISTER_NUM_LIMBS,
                RV64_REGISTER_AS,
                RV64_REGISTER_AS,
            );
            match lift_babybear(&register, 0x100, &extensions) {
                Some(LiftedInstr::Body(InstrAt {
                    instr: Instr::AluReg { op, rd, rs1, rs2 },
                    ..
                })) => assert_eq!((op, rd, rs1, rs2), (expected_op, 1, 2, 3)),
                other => panic!("unexpected lift: {other:?}"),
            }

            let immediate = alu_instruction(opcode, 0xff_ffff, RV64_REGISTER_AS, RV64_IMM_AS);
            assert!(lift_babybear(&immediate, 0x100, &extensions).is_none());
        }
    }

    #[test]
    fn load_store_address_space_domain() {
        let extensions = ExtensionRegistry::new();

        for opcode in [LOADD, LOADBU, LOADHU, LOADWU, LOADB, LOADH, LOADW] {
            let inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [8, 16, 0, 1, DEFERRAL_AS as usize, 1, 0],
            );

            assert!(lift_babybear(&inst, 0x100, &extensions).is_none());
        }

        for opcode in [STORED, STOREW, STOREH, STOREB] {
            let deferral_inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [8, 16, 0, 1, DEFERRAL_AS as usize, 1, 0],
            );
            assert!(lift_babybear(&deferral_inst, 0x100, &extensions).is_none());

            let public_values_inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [8, 16, 0, 1, PUBLIC_VALUES_AS as usize, 1, 0],
            );
            assert!(lift_babybear(&public_values_inst, 0x100, &extensions).is_none());
        }
    }

    #[test]
    fn load_store_requires_register_destination_address_space() {
        let extensions = ExtensionRegistry::new();

        for opcode in [LOADD, LOADBU, LOADHU, LOADWU, LOADB, LOADH, LOADW] {
            let inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    RV64_MEMORY_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            );

            assert!(lift_babybear(&inst, 0x100, &extensions).is_none());
        }

        for opcode in [STORED, STOREW, STOREH, STOREB] {
            let memory_inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    RV64_MEMORY_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            );
            assert!(lift_babybear(&memory_inst, 0x100, &extensions).is_none());

            let public_values_inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    RV64_MEMORY_AS as usize,
                    PUBLIC_VALUES_AS as usize,
                    1,
                    0,
                ],
            );
            assert!(lift_babybear(&public_values_inst, 0x100, &extensions).is_none());

            let valid_memory_inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                    1,
                    0,
                ],
            );
            assert!(lift_babybear(&valid_memory_inst, 0x100, &extensions).is_some());
        }
    }

    #[test]
    fn auipc_lifts_sign_extended_64_bit_value() {
        let pc = 0x1000;
        let insn = Instruction::<BabyBear>::from_usize(
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            [8, 0, 0x80_0000],
        );

        let lifted = lift_babybear(&insn, pc, &ExtensionRegistry::default());
        match lifted {
            Some(LiftedInstr::Body(InstrAt {
                pc: lifted_pc,
                instr: Instr::Auipc { rd, value },
                ..
            })) => {
                assert_eq!(lifted_pc, pc);
                assert_eq!(rd, 1);
                assert_eq!(value, 0xffff_ffff_8000_1000);
            }
            other => panic!("unexpected lift: {other:?}"),
        }
    }

    #[test]
    fn jal_preserves_negative_field_encoded_offset() {
        let pc = 0x1000;
        let insn = Instruction::<BabyBear>::from_isize(
            Rv64JalLuiOpcode::JAL.global_opcode(),
            8,
            0,
            -12,
            0,
            0,
        );

        let lifted = lift_babybear(&insn, pc, &ExtensionRegistry::default());
        match lifted {
            Some(LiftedInstr::Term {
                pc: lifted_pc,
                terminator: Terminator::Jump { link_rd, target },
                ..
            }) => {
                assert_eq!(lifted_pc, pc);
                assert_eq!(link_rd, Some(1));
                assert_eq!(target, pc - 12);
            }
            other => panic!("unexpected lift: {other:?}"),
        }
    }

    #[test]
    fn branch_preserves_negative_field_encoded_offset() {
        let pc = 0x1000;
        let insn = Instruction::<BabyBear>::from_isize(
            BranchEqualOpcode::BEQ.global_opcode(),
            8,
            16,
            -12,
            0,
            0,
        );

        let lifted = lift_babybear(&insn, pc, &ExtensionRegistry::default());
        match lifted {
            Some(LiftedInstr::Term {
                terminator: Terminator::Branch { target, .. },
                ..
            }) => assert_eq!(target, pc - 12),
            other => panic!("unexpected lift: {other:?}"),
        }
    }
}
