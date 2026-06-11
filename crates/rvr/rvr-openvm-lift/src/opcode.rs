//! OpenVM opcode -> rvr-openvm-ir mapping for RISC-V instructions.
//!
//! Supports the basic OpenVM ISA:
//! - RISC-V base instructions
//! - System instructions: TERMINATE, PHANTOM, PUBLISH
//! - System phantom sub-instructions: Nop, DebugPanic, CtStart, CtEnd
//! - STOREW e=2 dispatch (normal memory store)

use openvm_instructions::{
    instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode, SysPhantom, SystemOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BaseAluWOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode,
    DivRemWOpcode, LessThanOpcode, MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode,
    Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode, ShiftOpcode, ShiftWOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::AS_PUBLIC_VALUES;
use rvr_openvm_ir::{
    AluOp, BranchCond, Instr, InstrAt, LiftedInstr, MemWidth, MulDivOp, Terminator,
};

use crate::helpers::decode_imm_cg;

/// Lift a single OpenVM instruction to the new IR types.
///
/// Returns `None` for unrecognized opcodes. If a registered extension handles
/// the opcode, its `try_lift` is called.
pub fn lift_instruction<F: PrimeField32>(
    insn: &Instruction<F>,
    pc: u32,
    extensions: &crate::ExtensionRegistry<F>,
) -> Option<LiftedInstr> {
    let opcode = insn.opcode.as_usize();

    // System opcodes
    if opcode == SystemOpcode::TERMINATE.global_opcode_usize() {
        let exit_code = field_to_u32(insn.c);
        return Some(LiftedInstr::Term {
            pc,
            terminator: Terminator::Exit { code: exit_code },
            source_loc: None,
        });
    }
    if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
        let discriminant = (field_to_u32(insn.c) & 0xffff) as u16;
        if let Some(sys) = SysPhantom::from_repr(discriminant) {
            return Some(lift_phantom(sys, pc));
        }
        if let Some(lifted) = extensions.try_lift(insn, pc) {
            return Some(lifted);
        }
        return None;
    }

    // Decode the e field to determine R-type vs I-type
    let e = field_to_u32(insn.e);

    // BaseAlu: ADD=0x200, SUB=0x201, XOR=0x202, OR=0x203, AND=0x204, ADDW=0x270, SUBW=0x271
    if opcode == BaseAluOpcode::ADD.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Add));
    }
    if opcode == BaseAluOpcode::SUB.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Sub));
    }
    if opcode == BaseAluOpcode::XOR.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Xor));
    }
    if opcode == BaseAluOpcode::OR.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Or));
    }
    if opcode == BaseAluOpcode::AND.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::And));
    }
    if opcode == BaseAluWOpcode::ADDW.global_opcode_usize() {
        return Some(lift_alu_w(insn, pc, e, AluOp::Add));
    }
    if opcode == BaseAluWOpcode::SUBW.global_opcode_usize() {
        return Some(lift_alu_w(insn, pc, e, AluOp::Sub));
    }

    // Shift: SLL=0x205, SRL=0x206, SRA=0x207, SLLW=0x275, SRLW=0x276, SRAW=0x277
    if opcode == ShiftOpcode::SLL.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Sll));
    }
    if opcode == ShiftOpcode::SRL.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Srl));
    }
    if opcode == ShiftOpcode::SRA.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Sra));
    }
    if opcode == ShiftWOpcode::SLLW.global_opcode_usize() {
        return Some(lift_shift_w(insn, pc, e, AluOp::Sll));
    }
    if opcode == ShiftWOpcode::SRLW.global_opcode_usize() {
        return Some(lift_shift_w(insn, pc, e, AluOp::Srl));
    }
    if opcode == ShiftWOpcode::SRAW.global_opcode_usize() {
        return Some(lift_shift_w(insn, pc, e, AluOp::Sra));
    }

    // LessThan: SLT=0x208, SLTU=0x209
    if opcode == LessThanOpcode::SLT.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Slt));
    }
    if opcode == LessThanOpcode::SLTU.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Sltu));
    }

    // LoadStore: LOADD=0x210, LOADBU=0x211, LOADHU=0x212, LOADWU=0x213,
    //            STORED=0x214, STOREW=0x215, STOREH=0x216, STOREB=0x217,
    //            LOADB=0x218, LOADH=0x219, LOADW=0x21a
    if opcode == Rv64LoadStoreOpcode::LOADD.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Double, false));
    }
    if opcode == Rv64LoadStoreOpcode::LOADBU.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Byte, false));
    }
    if opcode == Rv64LoadStoreOpcode::LOADHU.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Half, false));
    }
    if opcode == Rv64LoadStoreOpcode::LOADWU.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Word, false));
    }
    if opcode == Rv64LoadStoreOpcode::LOADB.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Byte, true));
    }
    if opcode == Rv64LoadStoreOpcode::LOADH.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Half, true));
    }
    if opcode == Rv64LoadStoreOpcode::LOADW.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Word, true));
    }
    if opcode == Rv64LoadStoreOpcode::STORED.global_opcode_usize() {
        return Some(lift_store(insn, pc, MemWidth::Double));
    }
    if opcode == Rv64LoadStoreOpcode::STOREW.global_opcode_usize() {
        // e = RV64_MEMORY_AS is a normal store; e = AS_PUBLIC_VALUES is REVEAL,
        // handled by `Rv64IoExtension`.
        let addr_space = field_to_u32(insn.e);
        if addr_space != AS_PUBLIC_VALUES {
            return Some(lift_store(insn, pc, MemWidth::Word));
        }
    }
    if opcode == Rv64LoadStoreOpcode::STOREH.global_opcode_usize() {
        return Some(lift_store(insn, pc, MemWidth::Half));
    }
    if opcode == Rv64LoadStoreOpcode::STOREB.global_opcode_usize() {
        return Some(lift_store(insn, pc, MemWidth::Byte));
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

/// Convert a field element to u32.
pub fn field_to_u32<F: PrimeField32>(f: F) -> u32 {
    f.as_canonical_u32()
}

/// Convert a field element to i32 using modular arithmetic.
/// Values > p/2 are interpreted as negative (i.e. the field encoding of a negative integer).
pub fn field_to_i32<F: PrimeField32>(f: F) -> i32 {
    let v = f.as_canonical_u32();
    let p = F::ORDER_U32;
    if v > p / 2 {
        v.wrapping_sub(p) as i32
    } else {
        v as i32
    }
}

/// Decode register index from OpenVM operand (divided by RV64_REGISTER_NUM_LIMBS).
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (field_to_u32(f) / RV64_REGISTER_NUM_LIMBS as u32) as u8
}

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
pub fn body(pc: u32, instr: Instr) -> LiftedInstr {
    LiftedInstr::Body(InstrAt {
        pc,
        instr,
        source_loc: None,
    })
}

/// Helper to create a terminator instruction.
pub fn term(pc: u32, terminator: Terminator) -> LiftedInstr {
    LiftedInstr::Term {
        pc,
        terminator,
        source_loc: None,
    }
}

/// Lift ALU instruction (R-type when e!=0, I-type when e==0).
fn lift_alu<F: PrimeField32>(insn: &Instruction<F>, pc: u32, e: u32, op: AluOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);

    if e != 0 {
        // R-type: rs2 in c
        let rs2 = decode_reg(insn.c);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluReg { op, rd, rs1, rs2 })
    } else {
        // I-type: immediate in c as u24, sign-extend from 12 bits
        let raw_imm = field_to_u32(insn.c) & 0xffffff;
        let imm = sign_extend_12(raw_imm);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluImm { op, rd, rs1, imm })
    }
}

/// Lift shift instruction (R-type when e!=0, I-type shamt when e==0).
fn lift_shift<F: PrimeField32>(insn: &Instruction<F>, pc: u32, e: u32, op: AluOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);

    if e != 0 {
        // R-type shift: use AluReg (masking is done by the backend)
        let rs2 = decode_reg(insn.c);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluReg { op, rd, rs1, rs2 })
    } else {
        // I-type shamt: 6-bit for rv64 (registers are 64-bit wide)
        let shamt = (field_to_u32(insn.c) & 0x3f) as u8;
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::ShiftImm { op, rd, rs1, shamt })
    }
}

/// Lift MUL/DIV/REM R-type instruction.
fn lift_muldiv<F: PrimeField32>(insn: &Instruction<F>, pc: u32, op: MulDivOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let rs2 = decode_reg(insn.c);
    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::MulDiv { op, rd, rs1, rs2 })
}

/// Lift load instruction.
/// OpenVM encoding: rd=a/4, rs1=b/4, imm low16=c, sign=g!=0
fn lift_load<F: PrimeField32>(
    insn: &Instruction<F>,
    pc: u32,
    width: MemWidth,
    signed: bool,
) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let offset = decode_imm_cg(insn) as i16;

    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(
        pc,
        Instr::Load {
            width,
            signed,
            rd,
            rs1,
            offset,
        },
    )
}

/// Lift store instruction.
/// OpenVM encoding: rs2=a/4, rs1=b/4, imm low16=c, sign=g!=0
fn lift_store<F: PrimeField32>(insn: &Instruction<F>, pc: u32, width: MemWidth) -> LiftedInstr {
    let rs2 = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);
    let offset = decode_imm_cg(insn) as i16;

    body(
        pc,
        Instr::Store {
            width,
            rs1,
            rs2,
            offset,
        },
    )
}

/// Lift branch instruction.
/// OpenVM encoding: rs1=a/4, rs2=b/4, offset=c as signed (BabyBear modular)
fn lift_branch<F: PrimeField32>(insn: &Instruction<F>, pc: u32, cond: BranchCond) -> LiftedInstr {
    let rs1 = decode_reg(insn.a);
    let rs2 = decode_reg(insn.b);
    let offset = field_to_i32(insn.c);
    let target = (pc as i64 + offset as i64) as u32;

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
/// OpenVM encoding: rd=a/4, offset=c as signed (BabyBear modular)
fn lift_jal<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let offset = field_to_i32(insn.c);
    let target = (pc as i64 + offset as i64) as u32;

    let link_rd = if rd != 0 { Some(rd) } else { None };
    term(pc, Terminator::Jump { link_rd, target })
}

/// Lift JALR instruction.
/// OpenVM encoding: rd=a/4, rs1=b/4, imm low16=c, sign=g!=0
fn lift_jalr<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
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
/// OpenVM encoding: rd=a/4, upper20=c << 12
fn lift_lui<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let upper = field_to_u32(insn.c);
    let value = upper << 12;

    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::Lui { rd, value })
}

/// Lift AUIPC instruction.
/// OpenVM encoding: rd=a/4, upper20 = c << 8 (from rrs.rs: c = (imm & 0xfffff000) >> 8)
fn lift_auipc<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let shifted = field_to_u32(insn.c);
    let upper = shifted << 8; // reconstruct the upper 20 bits with low 12 zeros
    let value = pc.wrapping_add(upper);

    if rd == 0 {
        return body(pc, Instr::Nop);
    }
    body(pc, Instr::Auipc { rd, value })
}

/// Lift W-suffix ALU instruction (R-type when e!=0, I-type when e==0).
/// Result is the low 32 bits of the operation, sign-extended to 64 bits.
fn lift_alu_w<F: PrimeField32>(insn: &Instruction<F>, pc: u32, e: u32, op: AluOp) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);

    if e != 0 {
        let rs2 = decode_reg(insn.c);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluWReg { op, rd, rs1, rs2 })
    } else {
        let raw_imm = field_to_u32(insn.c) & 0xffffff;
        let imm = sign_extend_12(raw_imm);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluWImm { op, rd, rs1, imm })
    }
}

/// Lift W-suffix shift instruction (R-type when e!=0, I-type shamt when e==0).
/// Shamt is 5-bit (W shifts operate on 32-bit values regardless of register width).
fn lift_shift_w<F: PrimeField32>(
    insn: &Instruction<F>,
    pc: u32,
    e: u32,
    op: AluOp,
) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    let rs1 = decode_reg(insn.b);

    if e != 0 {
        let rs2 = decode_reg(insn.c);
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::AluWReg { op, rd, rs1, rs2 })
    } else {
        let shamt = (field_to_u32(insn.c) & 0x1f) as u8;
        if rd == 0 {
            return body(pc, Instr::Nop);
        }
        body(pc, Instr::ShiftWImm { op, rd, rs1, shamt })
    }
}

/// Lift W-suffix MUL/DIV/REM R-type instruction.
fn lift_muldiv_w<F: PrimeField32>(insn: &Instruction<F>, pc: u32, op: MulDivOp) -> LiftedInstr {
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
fn lift_phantom(sys: SysPhantom, pc: u32) -> LiftedInstr {
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
