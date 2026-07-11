//! C code generation for IR instructions and terminators.
use std::collections::HashSet;

use openvm_instructions::program::DEFAULT_PC_STEP;
use rvr_openvm_ir::*;

use super::context::{ArenaAlu3Baked, ArenaBranch2Baked, EmitContext};

/// Trait for instructions that can emit their own C code.
pub trait InstrCodegen {
    /// Emit the C body for this instruction into the buffer.
    /// `ctx` provides traced helpers for register/memory access.
    fn emit_c(&self, ctx: &mut EmitContext);
}

impl InstrCodegen for Instr {
    fn emit_c(&self, ctx: &mut EmitContext) {
        emit_instr(ctx, self);
    }
}

/// R4 arena-native geometry for one air's full (adapter + core) record.
/// Values are computed by the HOST (openvm-circuit / owning extension) from
/// the real record types via `size_of`/`align_of`/`offset_of!` and handed to
/// codegen, which bakes the offsets as literal stores in the generated
/// per-air emitter — the C side never mirrors the record structs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaNativeGeometry {
    pub adapter_size: usize,
    pub adapter_align: usize,
    pub core_size: usize,
    pub core_align: usize,
    /// Core-record byte offset within a Matrix row (the adapter trace width).
    pub core_off_matrix: usize,
    pub layout: ArenaNativeLayout,
}

impl ArenaNativeGeometry {
    pub fn core_off_dense(&self) -> usize {
        self.adapter_size.next_multiple_of(self.core_align)
    }

    pub fn stride_dense(&self) -> usize {
        (self.core_off_dense() + self.core_size).next_multiple_of(self.adapter_align)
    }
}

/// Per-shape field-offset tables (adapter offsets relative to the record
/// start; core offsets relative to the core record start, which sits at the
/// flavor's core offset carried in `ChipRecordBuf.core_off` at runtime).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaNativeLayout {
    Alu3(Alu3ArenaFieldOffsets),
    Branch2(Branch2ArenaFieldOffsets),
}

/// BabyBear modulus, for baking field-canonical immediates (negative branch
/// offsets encode as P + imm). rvr-openvm is OpenVM-specific, and the fused
/// byte-equality oracle guards this value against the host encoding.
pub const BABYBEAR_ORDER_U32: u32 = 0x7800_0001;

/// Offsets for the branch2-class full record: a two-read no-write branch
/// adapter + a comparison core with a/b operand limb arrays, a canonical
/// imm, and a `local_opcode` byte (BranchEqual and BranchLessThan share
/// this field shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Branch2ArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rs1_ptr: usize,
    pub rs2_ptr: usize,
    pub reads_aux0_prev_ts: usize,
    pub reads_aux1_prev_ts: usize,
    pub core_a: usize,
    pub core_b: usize,
    pub core_imm: usize,
    pub core_local_opcode: usize,
}

/// Offsets for the alu3-class full record: a two-read one-u16-block-write
/// adapter record + an alu core record with b/c operand limb arrays and a
/// `local_opcode` byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alu3ArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rd_ptr: usize,
    pub rs1_ptr: usize,
    pub rs2: usize,
    /// `usize::MAX` when the adapter has no rs2_as field (Mult adapter,
    /// whose rs2 is always a register pointer); the emitter skips the store.
    pub rs2_as: usize,
    /// `usize::MAX` when the adapter has no imm-sign field (byte adapter);
    /// the emitter then skips the store.
    pub rs2_imm_sign: usize,
    pub reads_aux0_prev_ts: usize,
    pub reads_aux1_prev_ts: usize,
    pub write_prev_ts: usize,
    pub write_prev_data: usize,
    pub core_b: usize,
    pub core_c: usize,
    /// `usize::MAX` when the family's core record has no local_opcode field
    /// (single-opcode airs like SRA); the emitter then skips the store.
    pub core_local_opcode: usize,
}

/// Compact inline-record wire shapes (R3). Each migrated instruction class
/// stores exactly its dynamic witness; the host re-derives program-redundant
/// operands from the instruction at `from_pc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineRecordShape {
    /// 2 reads + 1 write, single row (base ALU, Mul family): 44 bytes.
    Alu3,
    /// 2 reads, no write (branches): 32 bytes.
    Branch2,
    /// 1 (conditional) write (JalLui, Auipc): 20 bytes.
    Wr1,
    /// 1 read + 1 conditional write (Jalr): 32 bytes.
    Rw1,
}

/// The compact shape preflight codegen emits for a body instruction, or
/// `None` when the instruction stays on the verbose-log path. The single
/// source of truth shared by [`emit_instr`] and the host-side compile
/// metadata (`crates/vm` rvr `compile.rs`), which must skip the log assembler
/// for exactly these pcs.
pub fn inline_record_shape_for_instr(instr: &Instr) -> Option<InlineRecordShape> {
    match instr {
        Instr::AluReg { .. }
        | Instr::AluImm { .. }
        | Instr::ShiftImm { .. }
        | Instr::MulDiv { .. }
        | Instr::MulDivW { .. } => Some(InlineRecordShape::Alu3),
        Instr::Lui { .. } | Instr::Auipc { .. } => Some(InlineRecordShape::Wr1),
        // Main-memory loads/stores share the alu3 witness (rs1 value, block
        // read value / rs2 value, previous rd / block value). Public-values
        // stores (REVEAL) lift through the extension registry and stay on the
        // verbose-log path.
        Instr::Load { .. } | Instr::Store { .. } => Some(InlineRecordShape::Alu3),
        _ => None,
    }
}

/// The compact shape preflight codegen emits for a block terminator (see
/// [`inline_record_shape_for_instr`]).
pub fn inline_record_shape_for_terminator(term: &Terminator) -> Option<InlineRecordShape> {
    match term {
        Terminator::Branch { .. } => Some(InlineRecordShape::Branch2),
        Terminator::Jump { .. } => Some(InlineRecordShape::Wr1),
        Terminator::JumpDyn { .. } => Some(InlineRecordShape::Rw1),
        _ => None,
    }
}

/// Whether preflight codegen migrates this instruction to an inline compact
/// record (suppressing its memory-log entries).
pub fn instr_emits_inline_record(instr: &Instr) -> bool {
    inline_record_shape_for_instr(instr).is_some()
}

/// Emit C code for a body instruction.
pub fn emit_instr(ctx: &mut EmitContext, instr: &Instr) {
    match instr {
        Instr::AluReg { op, rd, rs1, rs2 } => {
            if ctx.inline_records_enabled() {
                // R4 baked operands for the AddSub family (local_opcode must
                // match BaseAluOpcode order: ADD=0, SUB=1; the fused oracle
                // byte-compares against the host assembler). Other AluReg ops
                // belong to airs without registered geometry, so `None`
                // keeps them on the compact wire regardless.
                // local_opcode per family enum order (AddSub: ADD=0/SUB=1;
                // LessThan: SLT=0/SLTU=1; ShiftLogical: SLL=0/SRL=1; SRA has
                // no local_opcode field — its layout carries the sentinel and
                // the baked value is ignored). Bitwise ops use the byte
                // adapter (different record) and stay compact for now.
                let reg_alu3 = |local_opcode: u8| {
                    Some(ArenaAlu3Baked {
                        rs2_field: (*rs2 as u32) * 8,
                        rs2_as: 1,
                        rs2_imm_sign: 0,
                        local_opcode,
                    })
                };
                let arena = match op {
                    AluOp::Add => reg_alu3(0),
                    AluOp::Sub => reg_alu3(1),
                    AluOp::Slt => reg_alu3(0),
                    AluOp::Sltu => reg_alu3(1),
                    AluOp::Sll => reg_alu3(0),
                    AluOp::Srl => reg_alu3(1),
                    AluOp::Sra => reg_alu3(0),
                    // Bitwise shares BaseAluOpcode's class offset.
                    AluOp::Xor => reg_alu3(2),
                    AluOp::Or => reg_alu3(3),
                    AluOp::And => reg_alu3(4),
                };
                ctx.emit_reg3_inline(*rd, *rs1, *rs2, arena, |l, r| alu_expr(*op, l, r));
            } else {
                let l = ctx.read_reg(*rs1);
                let r = ctx.read_reg(*rs2);
                ctx.write_reg(*rd, &alu_expr(*op, &l, &r));
            }
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            if ctx.inline_records_enabled() {
                // R4 imm-form baking (ADDI only; there is no SUB-imm). The
                // record's rs2 field is the 24-bit immediate encoding and
                // rs2_imm_sign its 12-bit sign, matching the host assembler.
                let imm_alu3 = |local_opcode: u8| {
                    Some(ArenaAlu3Baked {
                        rs2_field: (*imm as u32) & 0xFF_FFFF,
                        rs2_as: 0,
                        rs2_imm_sign: (*imm < 0) as u8,
                        local_opcode,
                    })
                };
                let arena = match op {
                    AluOp::Add => imm_alu3(0),
                    AluOp::Slt => imm_alu3(0),
                    AluOp::Sltu => imm_alu3(1),
                    AluOp::Xor => imm_alu3(2),
                    AluOp::Or => imm_alu3(3),
                    AluOp::And => imm_alu3(4),
                    _ => None,
                };
                ctx.emit_reg2imm_inline(*rd, *rs1, *imm as i64 as u64, arena, |l, v| {
                    alu_expr(*op, l, v)
                });
            } else {
                let l = ctx.read_reg(*rs1);
                ctx.trace_immediate();
                let r = imm_literal(*imm);
                ctx.write_reg(*rd, &alu_expr(*op, &l, &r));
            }
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            if ctx.inline_records_enabled() {
                // The record's c value is the raw shift amount; the result
                // expression uses the constant directly.
                // Shift-immediate: the record's rs2 field is the raw shamt
                // (positive, no sign), local_opcode per ShiftLogical order
                // (SRA's layout carries the no-field sentinel).
                let arena = match op {
                    AluOp::Sll => Some(0),
                    AluOp::Srl => Some(1),
                    AluOp::Sra => Some(0),
                    _ => None,
                }
                .map(|local_opcode| ArenaAlu3Baked {
                    rs2_field: u32::from(*shamt),
                    rs2_as: 0,
                    rs2_imm_sign: 0,
                    local_opcode,
                });
                ctx.emit_reg2imm_inline(*rd, *rs1, u64::from(*shamt), arena, |l, _| {
                    shift_imm_expr(*op, l, *shamt)
                });
            } else {
                let v = ctx.read_reg(*rs1);
                ctx.trace_immediate();
                ctx.write_reg(*rd, &shift_imm_expr(*op, &v, *shamt));
            }
        }
        Instr::Lui { rd, value } => {
            if ctx.inline_records_enabled() {
                ctx.emit_wr1_inline(Some(*rd), &sext32(*value));
            } else {
                ctx.write_reg(*rd, &sext32(*value));
            }
        }
        Instr::Auipc { rd, value } => {
            if ctx.inline_records_enabled() {
                ctx.emit_wr1_inline(Some(*rd), &hex_u64(*value));
            } else {
                ctx.write_reg(*rd, &hex_u64(*value));
            }
        }
        Instr::Load {
            width,
            signed,
            rd,
            rs1,
            offset,
        } => {
            if ctx.inline_records_enabled() {
                ctx.emit_load_inline(width.bytes(), *signed, *rd, *rs1, *offset);
            } else {
                let base = ctx.read_reg(*rs1);
                let val = ctx.read_mem(&base, *offset, width.bytes(), *signed);
                ctx.write_reg(*rd, &val);
            }
        }
        Instr::Store {
            width,
            rs1,
            rs2,
            offset,
        } => {
            if ctx.inline_records_enabled() {
                ctx.emit_store_inline(width.bytes(), *rs1, *rs2, *offset);
            } else {
                let base = ctx.read_reg(*rs1);
                let val = ctx.read_reg(*rs2);
                ctx.write_mem(&base, *offset, &val, width.bytes());
            }
        }
        Instr::AluWReg { op, rd, rs1, rs2 } => {
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            ctx.write_reg(*rd, &alu_w_expr(*op, &l, &r));
        }
        Instr::AluWImm { op, rd, rs1, imm } => {
            let l = ctx.read_reg(*rs1);
            ctx.trace_immediate();
            let r = imm_literal(*imm);
            ctx.write_reg(*rd, &alu_w_expr(*op, &l, &r));
        }
        Instr::ShiftWImm { op, rd, rs1, shamt } => {
            let v = ctx.read_reg(*rs1);
            ctx.trace_immediate();
            ctx.write_reg(*rd, &shift_w_imm_expr(*op, &v, *shamt));
        }
        Instr::MulDiv { op, rd, rs1, rs2 } => {
            emit_muldiv(ctx, *op, *rd, *rs1, *rs2);
        }
        Instr::MulDivW { op, rd, rs1, rs2 } => {
            emit_muldiv_w(ctx, *op, *rd, *rs1, *rs2);
        }

        // ── OpenVM system/IO instructions ────────────────────────────
        Instr::Nop => {
            ctx.trace_timestamp();
        }

        Instr::Ext(ext) => {
            ext.emit_c(ctx);
        }
    }
}

/// Context for terminator code generation (dispatch / tail-call info).
pub struct TermCtx<'a> {
    /// Set of valid block start PCs (for direct tail calls).
    pub valid_blocks: &'a HashSet<u64>,
}

/// Emit C code for a terminator using tail calls between blocks.
///
/// Static targets use direct tail calls: `return block_0x...(args);`
/// Dynamic targets go through the dispatch table: `return dispatch_table[idx](args);`
/// Exit/suspend/trap save hot regs to state and return to `rv_execute`.
pub fn emit_terminator(ctx: &mut EmitContext, term: &Terminator, pc: u64, tc: &TermCtx) {
    let next_pc = pc.wrapping_add(u64::from(DEFAULT_PC_STEP));
    let args = ctx.tail_call_args();
    match term {
        Terminator::FallThrough => {
            emit_tail_call(ctx, next_pc, &args, tc.valid_blocks);
        }
        Terminator::Jump { link_rd, target } => {
            if ctx.inline_records_enabled() {
                ctx.emit_wr1_inline(*link_rd, &hex_u64(next_pc));
            } else {
                ctx.write_reg(link_rd.unwrap_or(0), &hex_u64(next_pc));
            }
            emit_tail_call(ctx, *target, &args, tc.valid_blocks);
        }
        Terminator::JumpDyn {
            link_rd, rs1, imm, ..
        } => {
            let base = if ctx.inline_records_enabled() {
                ctx.emit_rw1_inline(*link_rd, *rs1, &hex_u64(next_pc))
            } else {
                let base = ctx.read_reg(*rs1);
                // Save base to a temporary when link_rd == rs1 to prevent
                // the link write from clobbering the jump target (e.g. jalr ra, ra, 0).
                let base = if link_rd.is_some_and(|rd| rd == *rs1) {
                    ctx.materialize_u64(&base)
                } else {
                    base
                };
                ctx.write_reg(link_rd.unwrap_or(0), &hex_u64(next_pc));
                base
            };
            let imm_val = *imm;
            let next_pc = if imm_val == 0 {
                ctx.materialize_u64(&format!("{base} & ~0x0000000000000001ull"))
            } else if imm_val > 0 {
                ctx.materialize_u64(&format!(
                    "({base} + {}) & ~0x0000000000000001ull",
                    hex_u64(imm_val as u64)
                ))
            } else {
                let abs = (-(imm_val as i64)) as u64;
                ctx.materialize_u64(&format!(
                    "({base} - {}) & ~0x0000000000000001ull",
                    hex_u64(abs)
                ))
            };
            ctx.write_line(&format!(
                "if (unlikely(!rv_pc_is_dispatchable({next_pc}))) {{"
            ));
            ctx.write_line(&format!("  [[clang::musttail]] return rv_trap({args});"));
            ctx.write_line("}");
            ctx.write_line(&format!("state->pc = {next_pc};"));
            ctx.write_line(&format!(
                "[[clang::musttail]] return dispatch_table[rv_dispatch_index({next_pc})]({args});"
            ));
        }
        Terminator::Branch {
            cond,
            rs1,
            rs2,
            target,
        } => {
            let (l, r) = if ctx.inline_records_enabled() {
                // R4 baked operands: register byte pointers, the
                // field-canonical branch immediate (P + imm for negative
                // offsets, matching the transpiler encoding the host
                // assembler reads back), and the family local_opcode
                // (BEQ=0/BNE=1; BLT=0/BLTU=1/BGE=2/BGEU=3).
                let imm = *target as i64 - pc as i64;
                let imm_canonical = if imm >= 0 {
                    imm as u32
                } else {
                    (i64::from(BABYBEAR_ORDER_U32) + imm) as u32
                };
                let local_opcode = match cond {
                    BranchCond::Eq => 0,
                    BranchCond::Ne => 1,
                    BranchCond::Lt => 0,
                    BranchCond::Ltu => 1,
                    BranchCond::Ge => 2,
                    BranchCond::Geu => 3,
                };
                let baked = ArenaBranch2Baked {
                    rs1_ptr: (*rs1 as u32) * 8,
                    rs2_ptr: (*rs2 as u32) * 8,
                    imm: imm_canonical,
                    local_opcode,
                };
                ctx.emit_branch2_inline(*rs1, *rs2, Some(baked))
            } else {
                (ctx.read_reg(*rs1), ctx.read_reg(*rs2))
            };
            let cmp = branch_cond_expr(*cond, &l, &r);
            let target_call = if tc.valid_blocks.contains(target) {
                format!("[[clang::musttail]] return block_0x{target:08x}({args});")
            } else {
                format!(
                    "[[clang::musttail]] return dispatch_table[rv_dispatch_index({})]({args});",
                    hex_u64(*target)
                )
            };
            ctx.write_line(&format!("if ({cmp}) {{"));
            ctx.write_line(&format!("  {target_call}"));
            ctx.write_line("}");
            emit_tail_call(ctx, next_pc, &args, tc.valid_blocks);
        }
        Terminator::Exit { code } => {
            ctx.sync_regs_to_state();
            ctx.write_line(&format!(
                "rv_set_status_at(state, {}, OPENVM_EXEC_TERMINATED, {code});",
                hex_u64(pc)
            ));
            ctx.write_line("return;");
        }
        Terminator::Trap { message } => {
            let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
            ctx.sync_regs_to_state();
            ctx.write_line(&format!(
                "rv_set_status_at(state, {}, OPENVM_EXEC_TRAPPED, 0);",
                hex_u64(pc)
            ));
            ctx.write_line(&format!("/* TRAP: {escaped} */"));
            ctx.write_line("return;");
        }
        Terminator::Extension(ext) => {
            let branch_to = |target: u64| -> String {
                if tc.valid_blocks.contains(&target) {
                    format!("[[clang::musttail]] return block_0x{target:08x}({args});")
                } else {
                    format!(
                        "[[clang::musttail]] return dispatch_table[rv_dispatch_index({})]({args});",
                        hex_u64(target)
                    )
                }
            };
            ext.emit_c_term(ctx, &branch_to);
        }
    }
}

/// Emit a tail call to a known PC. Uses a direct call if the target is a valid
/// block; otherwise falls back to the dispatch table.
fn emit_tail_call(ctx: &mut EmitContext, target: u64, args: &str, valid_blocks: &HashSet<u64>) {
    if valid_blocks.contains(&target) {
        ctx.write_line(&format!(
            "[[clang::musttail]] return block_0x{target:08x}({args});"
        ));
    } else {
        ctx.write_line(&format!(
            "[[clang::musttail]] return dispatch_table[rv_dispatch_index({})]({args});",
            hex_u64(target)
        ));
    }
}

// ── ALU helpers ──────────────────────────────────────────────────────────────

fn alu_expr(op: AluOp, left: &str, right: &str) -> String {
    match op {
        AluOp::Add => format!("{left} + {right}"),
        AluOp::Sub => format!("{left} - {right}"),
        AluOp::Sll => format!("{left} << ({right} & 0x3fu)"),
        AluOp::Slt => format!("((int64_t){left} < (int64_t){right}) ? 1u : 0u"),
        AluOp::Sltu => format!("({left} < {right}) ? 1u : 0u"),
        AluOp::Xor => format!("{left} ^ {right}"),
        AluOp::Srl => format!("{left} >> ({right} & 0x3fu)"),
        AluOp::Sra => format!("(uint64_t)((int64_t){left} >> ({right} & 0x3fu))"),
        AluOp::Or => format!("{left} | {right}"),
        AluOp::And => format!("{left} & {right}"),
    }
}

fn shift_imm_expr(op: AluOp, val: &str, shamt: u8) -> String {
    match op {
        AluOp::Sll => format!("{val} << {}", hex_u32(shamt as u32)),
        AluOp::Srl => format!("{val} >> {}", hex_u32(shamt as u32)),
        AluOp::Sra => format!("(uint64_t)((int64_t){val} >> {})", hex_u32(shamt as u32)),
        _ => unreachable!("invalid shift op {op:?}"),
    }
}

/// W-suffix ALU: low 32 bits, result sign-extended to 64.
fn alu_w_expr(op: AluOp, left: &str, right: &str) -> String {
    let inner = match op {
        AluOp::Add => format!("(uint32_t){left} + (uint32_t){right}"),
        AluOp::Sub => format!("(uint32_t){left} - (uint32_t){right}"),
        AluOp::Sll => format!("(uint32_t){left} << ((uint32_t){right} & 0x1fu)"),
        AluOp::Srl => format!("(uint32_t){left} >> ((uint32_t){right} & 0x1fu)"),
        AluOp::Sra => {
            format!("(uint32_t)((int32_t)(uint32_t){left} >> ((uint32_t){right} & 0x1fu))")
        }
        _ => unreachable!("no W variant for alu op {op:?}"),
    };
    format!("(uint64_t)(int32_t)({inner})")
}

/// W-suffix shift immediate: low 32 bits, result sign-extended to 64.
fn shift_w_imm_expr(op: AluOp, val: &str, shamt: u8) -> String {
    let inner = match op {
        AluOp::Sll => format!("(uint32_t){val} << {}", hex_u32(shamt as u32)),
        AluOp::Srl => format!("(uint32_t){val} >> {}", hex_u32(shamt as u32)),
        AluOp::Sra => format!(
            "(uint32_t)((int32_t)(uint32_t){val} >> {})",
            hex_u32(shamt as u32)
        ),
        _ => unreachable!("invalid W shift op {op:?}"),
    };
    format!("(uint64_t)(int32_t)({inner})")
}

/// Format an i32 immediate as a sign-extended 64-bit C literal.
fn imm_literal(imm: i32) -> String {
    sext32(imm as u32)
}

/// Sign-extend a u32 value to a 64-bit C literal (uint64_t).
fn sext32(value: u32) -> String {
    hex_u64(value as i32 as i64 as u64)
}

fn hex_u64(value: u64) -> String {
    format!("0x{value:016x}ull")
}

pub(super) fn hex_u32(value: u32) -> String {
    format!("0x{value:08x}u")
}

fn branch_cond_expr(cond: BranchCond, left: &str, right: &str) -> String {
    match cond {
        BranchCond::Eq => format!("{left} == {right}"),
        BranchCond::Ne => format!("{left} != {right}"),
        BranchCond::Lt => format!("(int64_t){left} < (int64_t){right}"),
        BranchCond::Ge => format!("(int64_t){left} >= (int64_t){right}"),
        BranchCond::Ltu => format!("{left} < {right}"),
        BranchCond::Geu => format!("{left} >= {right}"),
    }
}

// ── MulDiv ──────────────────────────────────────────────────────────────────

/// C expression computing a MulDiv result from two read register values.
/// Division/remainder zero-divisor and overflow semantics live inside the
/// `rv_*` helpers, so the inline-record path reuses these expressions
/// verbatim.
fn muldiv_expr(op: MulDivOp, l: &str, r: &str) -> String {
    match op {
        MulDivOp::Mul => format!("{l} * {r}"),
        MulDivOp::Mulh => format!("rv_mulh((int64_t){l}, (int64_t){r})"),
        MulDivOp::Mulhsu => format!("rv_mulhsu((int64_t){l}, {r})"),
        MulDivOp::Mulhu => format!("rv_mulhu({l}, {r})"),
        MulDivOp::Div => format!("rv_div((int64_t){l}, (int64_t){r})"),
        MulDivOp::Divu => format!("rv_divu({l}, {r})"),
        MulDivOp::Rem => format!("rv_rem((int64_t){l}, (int64_t){r})"),
        MulDivOp::Remu => format!("rv_remu({l}, {r})"),
    }
}

fn emit_muldiv(ctx: &mut EmitContext, op: MulDivOp, rd: u8, rs1: u8, rs2: u8) {
    if ctx.inline_records_enabled() {
        // Baked consts per air: Mul core has no local_opcode (sentinel
        // layout); MulH: MULH=0/MULHSU=1/MULHU=2; DivRem: DIV=0/DIVU=1/
        // REM=2/REMU=3. The Mult adapter's rs2 is always a register
        // pointer (rs2_as carries the layout sentinel).
        let local_opcode = match op {
            MulDivOp::Mul => 0,
            MulDivOp::Mulh => 0,
            MulDivOp::Mulhsu => 1,
            MulDivOp::Mulhu => 2,
            MulDivOp::Div => 0,
            MulDivOp::Divu => 1,
            MulDivOp::Rem => 2,
            MulDivOp::Remu => 3,
        };
        let arena = Some(ArenaAlu3Baked {
            rs2_field: (rs2 as u32) * 8,
            rs2_as: 0,
            rs2_imm_sign: 0,
            local_opcode,
        });
        ctx.emit_reg3_inline(rd, rs1, rs2, arena, |l, r| muldiv_expr(op, l, r));
        return;
    }
    let l = ctx.read_reg(rs1);
    let r = ctx.read_reg(rs2);
    ctx.write_reg(rd, &muldiv_expr(op, &l, &r));
}

/// C expression computing a MulDivW result (see [`muldiv_expr`]).
fn muldiv_w_expr(op: MulDivOp, l: &str, r: &str) -> String {
    match op {
        MulDivOp::Mul => format!("(uint64_t)(int32_t)((uint32_t){l} * (uint32_t){r})"),
        MulDivOp::Div => format!("rv_divw((int32_t)(uint32_t){l}, (int32_t)(uint32_t){r})"),
        MulDivOp::Divu => format!("rv_divuw((uint32_t){l}, (uint32_t){r})"),
        MulDivOp::Rem => format!("rv_remw((int32_t)(uint32_t){l}, (int32_t)(uint32_t){r})"),
        MulDivOp::Remu => format!("rv_remuw((uint32_t){l}, (uint32_t){r})"),
        _ => unreachable!("no W variant for mul op {op:?}"),
    }
}

fn emit_muldiv_w(ctx: &mut EmitContext, op: MulDivOp, rd: u8, rs1: u8, rs2: u8) {
    if ctx.inline_records_enabled() {
        ctx.emit_reg3_inline(rd, rs1, rs2, None, |l, r| muldiv_w_expr(op, l, r));
        return;
    }
    let l = ctx.read_reg(rs1);
    let r = ctx.read_reg(rs2);
    ctx.write_reg(rd, &muldiv_w_expr(op, &l, &r));
}
