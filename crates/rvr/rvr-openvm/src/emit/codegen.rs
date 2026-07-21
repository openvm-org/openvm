//! C code generation for IR instructions and terminators.
use std::collections::HashSet;

use openvm_instructions::program::DEFAULT_PC_STEP;
use rvr_openvm_ir::*;

use super::context::EmitContext;

/// Trait for instructions that can emit their own C code.
pub trait InstrCodegen {
    /// Emit the C body for this instruction into the buffer.
    /// `ctx` handles register and memory access for the current execution mode.
    fn emit_c(&self, ctx: &mut EmitContext);
}

impl InstrCodegen for Instr {
    fn emit_c(&self, ctx: &mut EmitContext) {
        emit_instr(ctx, self);
    }
}

/// Emit C code for a body instruction.
pub fn emit_instr(ctx: &mut EmitContext, instr: &Instr) {
    match instr {
        Instr::AluReg { op, rd, rs1, rs2 } => {
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            let value = constant_alu_reg_result(*op, *rs1, *rs2)
                .map(hex_u64)
                .unwrap_or_else(|| alu_expr(*op, &l, &r));
            ctx.write_reg(*rd, &value);
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            let l = ctx.read_reg(*rs1);
            let r = imm_literal(*imm);
            let value = constant_alu_imm_result(*op, *rs1, *imm)
                .map(hex_u64)
                .unwrap_or_else(|| alu_expr(*op, &l, &r));
            ctx.write_reg(*rd, &value);
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            let v = ctx.read_reg(*rs1);
            ctx.write_reg(*rd, &shift_imm_expr(*op, &v, *shamt));
        }
        Instr::Lui { rd, value } => {
            ctx.write_reg(*rd, &sext32(*value));
        }
        Instr::Auipc { rd, value } => {
            ctx.write_reg(*rd, &hex_u64(*value));
        }
        Instr::Load {
            width,
            signed,
            rd,
            rs1,
            offset,
        } => {
            let base = ctx.read_reg(*rs1);
            let val = ctx.read_mem(&base, *offset, width.bytes(), *signed);
            ctx.write_reg(*rd, &val);
        }
        Instr::Store {
            width,
            rs1,
            rs2,
            offset,
        } => {
            let base = ctx.read_reg(*rs1);
            let val = ctx.read_reg(*rs2);
            ctx.write_mem(&base, *offset, &val, width.bytes());
        }
        Instr::AluWReg { op, rd, rs1, rs2 } => {
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            ctx.write_reg(*rd, &alu_w_expr(*op, &l, &r));
        }
        Instr::AluWImm { op, rd, rs1, imm } => {
            let l = ctx.read_reg(*rs1);
            let r = imm_literal(*imm);
            ctx.write_reg(*rd, &alu_w_expr(*op, &l, &r));
        }
        Instr::ShiftWImm { op, rd, rs1, shamt } => {
            let v = ctx.read_reg(*rs1);
            ctx.write_reg(*rd, &shift_w_imm_expr(*op, &v, *shamt));
        }
        Instr::MulDiv { op, rd, rs1, rs2 } => {
            emit_muldiv(ctx, *op, *rd, *rs1, *rs2);
        }
        Instr::MulDivW { op, rd, rs1, rs2 } => {
            emit_muldiv_w(ctx, *op, *rd, *rs1, *rs2);
        }

        // ── OpenVM system/IO instructions ────────────────────────────
        Instr::Nop => {}

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
/// Static targets use direct tail calls or trap when no block exists.
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
            if let Some(rd) = link_rd {
                ctx.write_reg(*rd, &hex_u64(next_pc));
            }
            emit_tail_call(ctx, *target, &args, tc.valid_blocks);
        }
        Terminator::JumpDyn {
            link_rd, rs1, imm, ..
        } => {
            let base = ctx.read_reg(*rs1);
            // Save base to a temporary when link_rd == rs1 to prevent
            // the link write from clobbering the jump target (e.g. jalr ra, ra, 0).
            let base = if link_rd.is_some_and(|rd| rd == *rs1) {
                ctx.materialize_u64(&base)
            } else {
                base
            };
            if let Some(rd) = link_rd {
                ctx.write_reg(*rd, &hex_u64(next_pc));
            }
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
            if let Some(taken) = constant_branch_result(*cond, *rs1, *rs2) {
                if ctx.traces_values() {
                    ctx.read_reg(*rs1);
                    ctx.read_reg(*rs2);
                }
                let destination = if taken { *target } else { next_pc };
                emit_tail_call(ctx, destination, &args, tc.valid_blocks);
                return;
            }
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            let cmp = branch_cond_expr(*cond, &l, &r);
            let target_call = static_tail_call(*target, &args, tc.valid_blocks);
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
            let branch_to = |target| static_tail_call(target, &args, tc.valid_blocks);
            ext.emit_c_term(ctx, &branch_to);
        }
    }
}

fn static_tail_call(target: u64, args: &str, valid_blocks: &HashSet<u64>) -> String {
    if valid_blocks.contains(&target) {
        format!("[[clang::musttail]] return block_0x{target:08x}({args});")
    } else {
        format!("state->pc = 0x{target:016x}ull; [[clang::musttail]] return rv_trap({args});")
    }
}

/// Emit a tail call to a statically known PC.
fn emit_tail_call(ctx: &mut EmitContext, target: u64, args: &str, valid_blocks: &HashSet<u64>) {
    ctx.write_line(&static_tail_call(target, args, valid_blocks));
}

// ── ALU helpers ──────────────────────────────────────────────────────────────

fn alu_expr(op: AluOp, left: &str, right: &str) -> String {
    match op {
        AluOp::Add => format!("{left} + {right}"),
        AluOp::Sub => format!("{left} - {right}"),
        AluOp::Sll => format!("{left} << ({right} & 0x3fu)"),
        AluOp::Slt => format!("(int64_t){left} < (int64_t){right}"),
        AluOp::Sltu => format!("{left} < {right}"),
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

fn constant_branch_result(cond: BranchCond, rs1: u8, rs2: u8) -> Option<bool> {
    if rs1 == rs2 {
        return Some(matches!(
            cond,
            BranchCond::Eq | BranchCond::Ge | BranchCond::Geu
        ));
    }
    match cond {
        BranchCond::Ltu if rs2 == 0 => Some(false),
        BranchCond::Geu if rs2 == 0 => Some(true),
        _ => None,
    }
}

fn constant_alu_reg_result(op: AluOp, rs1: u8, rs2: u8) -> Option<u64> {
    match op {
        AluOp::Slt | AluOp::Sltu if rs1 == rs2 => Some(0),
        AluOp::Sltu if rs2 == 0 => Some(0),
        _ => None,
    }
}

fn constant_alu_imm_result(op: AluOp, rs1: u8, imm: i32) -> Option<u64> {
    match op {
        AluOp::Slt if rs1 == 0 => Some(u64::from(0 < imm)),
        AluOp::Sltu if rs1 == 0 => Some(u64::from(imm != 0)),
        AluOp::Sltu if imm == 0 => Some(0),
        _ => None,
    }
}

// ── MulDiv ──────────────────────────────────────────────────────────────────

fn emit_muldiv(ctx: &mut EmitContext, op: MulDivOp, rd: u8, rs1: u8, rs2: u8) {
    let l = ctx.read_reg(rs1);
    let r = ctx.read_reg(rs2);

    match op {
        MulDivOp::Mul => {
            ctx.write_reg(rd, &format!("{l} * {r}"));
        }
        MulDivOp::Mulh => {
            ctx.write_reg(rd, &format!("rv_mulh((int64_t){l}, (int64_t){r})"));
        }
        MulDivOp::Mulhsu => {
            ctx.write_reg(rd, &format!("rv_mulhsu((int64_t){l}, {r})"));
        }
        MulDivOp::Mulhu => {
            ctx.write_reg(rd, &format!("rv_mulhu({l}, {r})"));
        }
        MulDivOp::Div => {
            ctx.write_reg(rd, &format!("rv_div((int64_t){l}, (int64_t){r})"));
        }
        MulDivOp::Divu => {
            ctx.write_reg(rd, &format!("rv_divu({l}, {r})"));
        }
        MulDivOp::Rem => {
            ctx.write_reg(rd, &format!("rv_rem((int64_t){l}, (int64_t){r})"));
        }
        MulDivOp::Remu => {
            ctx.write_reg(rd, &format!("rv_remu({l}, {r})"));
        }
    }
}

fn emit_muldiv_w(ctx: &mut EmitContext, op: MulDivOp, rd: u8, rs1: u8, rs2: u8) {
    let l = ctx.read_reg(rs1);
    let r = ctx.read_reg(rs2);
    match op {
        MulDivOp::Mul => {
            ctx.write_reg(
                rd,
                &format!("(uint64_t)(int32_t)((uint32_t){l} * (uint32_t){r})"),
            );
        }
        MulDivOp::Div => {
            ctx.write_reg(
                rd,
                &format!("rv_divw((int32_t)(uint32_t){l}, (int32_t)(uint32_t){r})"),
            );
        }
        MulDivOp::Divu => {
            ctx.write_reg(rd, &format!("rv_divuw((uint32_t){l}, (uint32_t){r})"));
        }
        MulDivOp::Rem => {
            ctx.write_reg(
                rd,
                &format!("rv_remw((int32_t)(uint32_t){l}, (int32_t)(uint32_t){r})"),
            );
        }
        MulDivOp::Remu => {
            ctx.write_reg(rd, &format!("rv_remuw((uint32_t){l}, (uint32_t){r})"));
        }
        _ => unreachable!("no W variant for mul op {op:?}"),
    }
}
