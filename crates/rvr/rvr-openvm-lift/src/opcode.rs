//! OpenVM opcode -> rvr-openvm-ir mapping for each RV32IM instruction.
//!
//! Supports the basic OpenVM ISA:
//! - RV32IM base instructions
//! - System instructions: TERMINATE, PHANTOM, PUBLISH
//! - Phantom sub-instructions: Nop, DebugPanic, CtStart, CtEnd, Rv32HintInput, Rv32PrintStr,
//!   Rv32HintRandom
//! - IO instructions: HINT_STOREW, HINT_BUFFER
//! - STOREW address space dispatch: memory (e=2), reveal (e=3)

use openvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode, SystemOpcode,
};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv32AuipcOpcode, Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode,
    Rv32LoadStoreOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{
    AluOp, BranchCond, Instr, InstrAt, LiftedInstr, MemWidth, MulDivOp, Terminator,
};

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
        let discriminant = field_to_u32(insn.c) & 0xffff;
        // Try built-in phantom handlers first, then fall through to extensions
        // for unknown discriminants (e.g. algebra HintNonQr/HintSqrt).
        match discriminant {
            phantom_disc::NOP
            | phantom_disc::CT_START
            | phantom_disc::CT_END
            | phantom_disc::DEBUG_PANIC
            | phantom_disc::RV32_HINT_INPUT
            | phantom_disc::RV32_PRINT_STR
            | phantom_disc::RV32_HINT_RANDOM => return Some(lift_phantom(insn, pc)),
            _ => {
                if let Some(lifted) = extensions.try_lift(insn, pc) {
                    return Some(lifted);
                }
                return Some(body(pc, Instr::Nop));
            }
        }
    }

    // HINT_STOREW: _,b,_,1,2 — pop 4 bytes from hint stream to mem[[b]_1]
    if opcode == Rv32HintStoreOpcode::HINT_STOREW.global_opcode_usize() {
        return Some(lift_hint_storew(insn, pc));
    }
    // HINT_BUFFER: a,b,_,1,2 — pop 4*[a]_1 bytes from hint stream to mem[[b]_1]
    if opcode == Rv32HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
        return Some(lift_hint_buffer(insn, pc));
    }

    // Decode the e field to determine R-type vs I-type
    let e = field_to_u32(insn.e);

    // BaseAlu: ADD=0x200, SUB=0x201, XOR=0x202, OR=0x203, AND=0x204
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

    // Shift: SLL=0x205, SRL=0x206, SRA=0x207
    if opcode == ShiftOpcode::SLL.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Sll));
    }
    if opcode == ShiftOpcode::SRL.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Srl));
    }
    if opcode == ShiftOpcode::SRA.global_opcode_usize() {
        return Some(lift_shift(insn, pc, e, AluOp::Sra));
    }

    // LessThan: SLT=0x208, SLTU=0x209
    if opcode == LessThanOpcode::SLT.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Slt));
    }
    if opcode == LessThanOpcode::SLTU.global_opcode_usize() {
        return Some(lift_alu(insn, pc, e, AluOp::Sltu));
    }

    // LoadStore: LOADW=0x210, LOADBU=0x211, LOADHU=0x212, STOREW=0x213,
    //            STOREH=0x214, STOREB=0x215, LOADB=0x216, LOADH=0x217
    if opcode == Rv32LoadStoreOpcode::LOADW.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Word, false));
    }
    if opcode == Rv32LoadStoreOpcode::LOADBU.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Byte, false));
    }
    if opcode == Rv32LoadStoreOpcode::LOADHU.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Half, false));
    }
    if opcode == Rv32LoadStoreOpcode::LOADB.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Byte, true));
    }
    if opcode == Rv32LoadStoreOpcode::LOADH.global_opcode_usize() {
        return Some(lift_load(insn, pc, MemWidth::Half, true));
    }
    if opcode == Rv32LoadStoreOpcode::STOREW.global_opcode_usize() {
        // Dispatch based on write address space (field e):
        //   2 = main memory (normal store)
        //   3 = user IO / reveal (public outputs)
        let addr_space = field_to_u32(insn.e);
        return match addr_space {
            3 => Some(lift_reveal(insn, pc)),
            _ => Some(lift_store(insn, pc, MemWidth::Word)),
        };
    }
    if opcode == Rv32LoadStoreOpcode::STOREH.global_opcode_usize() {
        return Some(lift_store(insn, pc, MemWidth::Half));
    }
    if opcode == Rv32LoadStoreOpcode::STOREB.global_opcode_usize() {
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
    if opcode == Rv32JalLuiOpcode::JAL.global_opcode_usize() {
        return Some(lift_jal(insn, pc));
    }
    // LUI=0x231
    if opcode == Rv32JalLuiOpcode::LUI.global_opcode_usize() {
        return Some(lift_lui(insn, pc));
    }
    // JALR=0x235
    if opcode == Rv32JalrOpcode::JALR.global_opcode_usize() {
        return Some(lift_jalr(insn, pc));
    }
    // AUIPC=0x240
    if opcode == Rv32AuipcOpcode::AUIPC.global_opcode_usize() {
        return Some(lift_auipc(insn, pc));
    }

    // MUL=0x250
    if opcode == MulOpcode::MUL.global_opcode_usize() {
        return Some(lift_muldiv(insn, pc, MulDivOp::Mul));
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

/// Decode register index from OpenVM operand (divided by RV32_REGISTER_NUM_LIMBS).
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (field_to_u32(f) / RV32_REGISTER_NUM_LIMBS as u32) as u8
}

/// Decode an immediate from the (c, g) field pair used by JALR, LOAD, and STORE.
///
/// OpenVM stores the lower 16 bits of the sign-extended immediate in `c`,
/// and the sign bit in `g`. The full 32-bit value is reconstructed as:
///   imm = (c & 0xFFFF) + g * 0xFFFF0000
fn decode_imm_cg<F: PrimeField32>(insn: &Instruction<F>) -> u32 {
    let low16 = field_to_u32(insn.c) & 0xffff;
    let is_neg = field_to_u32(insn.g) != 0;
    low16.wrapping_add(if is_neg { 0xFFFF0000 } else { 0 })
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
        // I-type shamt: immediate in c (already just the shamt bits)
        let shamt = (field_to_u32(insn.c) & 0x1f) as u8;
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

// ============= System / IO Instructions =============

/// Phantom sub-instruction discriminants (lower 16 bits of field c).
mod phantom_disc {
    // System phantoms
    pub const NOP: u32 = 0x00;
    pub const DEBUG_PANIC: u32 = 0x01;
    pub const CT_START: u32 = 0x02;
    pub const CT_END: u32 = 0x03;
    // RV32IM extension phantoms
    pub const RV32_HINT_INPUT: u32 = 0x20;
    pub const RV32_PRINT_STR: u32 = 0x21;
    pub const RV32_HINT_RANDOM: u32 = 0x22;
}

/// Lift PHANTOM instruction by dispatching on the sub-discriminant in field c.
fn lift_phantom<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let c_val = field_to_u32(insn.c);
    let discriminant = c_val & 0xffff;

    match discriminant {
        // Nop, CtStart, CtEnd — no-ops for execution
        phantom_disc::NOP | phantom_disc::CT_START | phantom_disc::CT_END => body(pc, Instr::Nop),

        // DebugPanic — trap on host
        phantom_disc::DEBUG_PANIC => term(
            pc,
            Terminator::Trap {
                message: "PHANTOM DebugPanic".to_string(),
            },
        ),

        // HintInput — pop from input_stream, reset hint_stream with length-prefixed data
        phantom_disc::RV32_HINT_INPUT => body(pc, Instr::HintInput),

        // PrintStr — print UTF-8 string from memory; a=ptr_reg, b=len_reg
        phantom_disc::RV32_PRINT_STR => {
            let ptr_reg = decode_reg(insn.a);
            let len_reg = decode_reg(insn.b);
            body(pc, Instr::PrintStr { ptr_reg, len_reg })
        }

        // HintRandom — fill hint_stream with [a]_1 random words
        phantom_disc::RV32_HINT_RANDOM => {
            let num_words_reg = decode_reg(insn.a);
            body(pc, Instr::HintRandom { num_words_reg })
        }

        // Unknown phantom — treat as nop (forward compatible)
        _ => body(pc, Instr::Nop),
    }
}

/// Lift HINT_STOREW: pop 4 bytes from hint_stream, write to mem[ptr].
fn lift_hint_storew<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let ptr_reg = decode_reg(insn.b);
    body(pc, Instr::HintStoreW { ptr_reg })
}

/// Lift HINT_BUFFER: pop num_words*4 bytes from hint_stream, write sequentially.
fn lift_hint_buffer<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let num_words_reg = decode_reg(insn.a);
    let ptr_reg = decode_reg(insn.b);
    body(
        pc,
        Instr::HintBuffer {
            ptr_reg,
            num_words_reg,
        },
    )
}

/// Lift REVEAL (STOREW with e=3): write register value to user IO address space.
fn lift_reveal<F: PrimeField32>(insn: &Instruction<F>, pc: u32) -> LiftedInstr {
    let src_reg = decode_reg(insn.a);
    let ptr_reg = decode_reg(insn.b);
    let offset = decode_imm_cg(insn);
    body(
        pc,
        Instr::Reveal {
            src_reg,
            ptr_reg,
            offset,
        },
    )
}
