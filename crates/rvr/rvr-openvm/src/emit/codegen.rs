//! C code generation for IR instructions and terminators.

use openvm_circuit::system::memory::merkle::public_values::PUBLIC_VALUES_AS;
use openvm_instructions::riscv::RV32_MEMORY_AS;
use rvr_openvm_ir::*;

use super::context::EmitContext;

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

/// Emit C code for a body instruction.
pub fn emit_instr(ctx: &mut EmitContext, instr: &Instr) {
    match instr {
        Instr::AluReg { op, rd, rs1, rs2 } => {
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            ctx.write_reg(*rd, &alu_expr(*op, &l, &r));
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            let l = ctx.read_reg(*rs1);
            let r = imm_literal(*imm);
            ctx.write_reg(*rd, &alu_expr(*op, &l, &r));
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            let v = ctx.read_reg(*rs1);
            ctx.write_reg(*rd, &shift_imm_expr(*op, &v, *shamt));
        }
        Instr::Lui { rd, value } => {
            ctx.write_reg(*rd, &hex_u32(*value));
        }
        Instr::Auipc { rd, value } => {
            ctx.write_reg(*rd, &hex_u32(*value));
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
        Instr::MulDiv { op, rd, rs1, rs2 } => {
            emit_muldiv(ctx, *op, *rd, *rs1, *rs2);
        }

        // ── OpenVM system/IO instructions ────────────────────────────
        Instr::Nop => {}

        Instr::HintInput => {
            ctx.extern_call("openvm_hint_input", &[]);
        }

        // Phantom instructions: use read_reg_raw (no trace).
        Instr::PrintStr { ptr_reg, len_reg } => {
            let ptr = ctx.read_reg_raw(*ptr_reg);
            let len = ctx.read_reg_raw(*len_reg);
            ctx.extern_call("openvm_print_str", &[&ptr, &len]);
        }
        Instr::HintRandom { num_words_reg } => {
            let n = ctx.read_reg_raw(*num_words_reg);
            ctx.extern_call("openvm_hint_random", &[&n]);
        }

        // Memory-writing IO: traced register reads + trace-only mem access.
        Instr::HintStoreW { ptr_reg } => {
            let ptr = ctx.read_reg(*ptr_reg);
            ctx.trace_mem_access(&ptr, RV32_MEMORY_AS);
            ctx.extern_call("openvm_hint_storew", &[&ptr]);
        }
        Instr::HintBuffer {
            ptr_reg,
            num_words_reg,
        } => {
            let ptr = ctx.read_reg(*ptr_reg);
            let n = ctx.read_reg(*num_words_reg);
            let chip = ctx.hint_store_chip_idx;
            // OpenVM's HINT_BUFFER executor adds `num_words` rows to its chip
            // in one shot. We split that into two pieces:
            //   - The block-entry chip accounting in `project.rs` already
            //     credited +1 to this PC's chip (the static per-instruction
            //     contribution, like every other opcode).
            //   - Here we add the remaining +(n - 1) at runtime, since `n` is
            //     a register value not known at codegen time.
            // The `n > 1` guard skips the call when the static +1 is already
            // the whole answer (HINT_STOREW-shaped hints).
            if chip != u32::MAX {
                ctx.write_line(&format!("if ({n} > 1) {{"));
                ctx.write_line(&format!("  trace_chip(state, {chip}u, {n} - 1);"));
                ctx.write_line("}");
            }
            // TODO: change to trace_rd_mem_u32_range
            ctx.write_line(&format!("if ({n} > 0) {{"));
            ctx.write_line(&format!(
                "  trace_mem_access_u32_range(state, {ptr}, {n}, {RV32_MEMORY_AS}u);"
            ));
            ctx.write_line("}");
            ctx.extern_call("openvm_hint_buffer", &[&ptr, &n]);
        }
        Instr::Reveal {
            src_reg,
            ptr_reg,
            offset,
        } => {
            let src = ctx.read_reg(*src_reg);
            let ptr = ctx.read_reg(*ptr_reg);
            ctx.trace_mem_access(&ptr, PUBLIC_VALUES_AS);
            let off_s = hex_u32(*offset);
            ctx.extern_call("openvm_reveal", &[&src, &ptr, &off_s]);
        }
        Instr::Ext(ext) => {
            ext.emit_c(ctx);
        }
    }
}

/// Context for terminator code generation (dispatch / tail-call info).
pub struct TermCtx<'a> {
    /// Set of valid block start PCs (for direct tail calls).
    pub valid_blocks: &'a std::collections::HashSet<u32>,
}

/// Emit C code for a terminator using tail calls between blocks.
///
/// Static targets use direct tail calls: `return block_0x...(args);`
/// Dynamic targets go through the dispatch table: `return dispatch_table[idx](args);`
/// Exit/suspend/trap save hot regs to state and return to `rv_execute`.
pub fn emit_terminator(ctx: &mut EmitContext, term: &Terminator, pc: u32, tc: &TermCtx) {
    let next_pc = pc.wrapping_add(4);
    let args = ctx.tail_call_args();
    match term {
        Terminator::FallThrough => {
            emit_tail_call(ctx, next_pc, &args, tc.valid_blocks);
        }
        Terminator::Jump { link_rd, target } => {
            if let Some(rd) = link_rd {
                ctx.write_reg(*rd, &hex_u32(next_pc));
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
                ctx.materialize_u32(&base)
            } else {
                base
            };
            if let Some(rd) = link_rd {
                ctx.write_reg(*rd, &hex_u32(next_pc));
            }
            let imm_val = *imm;
            let next_pc = if imm_val == 0 {
                ctx.materialize_u32(&format!("{base} & ~0x00000001u"))
            } else if imm_val > 0 {
                ctx.materialize_u32(&format!(
                    "({base} + {}) & ~0x00000001u",
                    hex_u32(imm_val as u32)
                ))
            } else {
                let abs = (-(imm_val as i64)) as u32;
                ctx.materialize_u32(&format!("({base} - {}) & ~0x00000001u", hex_u32(abs)))
            };
            ctx.write_line(&format!("state->pc = {next_pc};"));
            ctx.write_line(&format!(
                "[[clang::musttail]] return dispatch_table[dispatch_index({next_pc})]({args});"
            ));
        }
        Terminator::Branch {
            cond,
            rs1,
            rs2,
            target,
        } => {
            let l = ctx.read_reg(*rs1);
            let r = ctx.read_reg(*rs2);
            let cmp = branch_cond_expr(*cond, &l, &r);
            let target_call = if tc.valid_blocks.contains(target) {
                format!("[[clang::musttail]] return block_0x{target:08x}({args});")
            } else {
                format!(
                    "[[clang::musttail]] return dispatch_table[dispatch_index({})]({args});",
                    hex_u32(*target)
                )
            };
            ctx.write_line(&format!("if ({cmp}) {{"));
            ctx.write_line(&format!("  {target_call}"));
            ctx.write_line("}");
            emit_tail_call(ctx, next_pc, &args, tc.valid_blocks);
        }
        Terminator::Exit { code } => {
            ctx.sync_regs_to_state();
            ctx.write_line(&format!("state->pc = {};", hex_u32(pc)));
            ctx.write_line("state->has_exited = OPENVM_EXEC_TERMINATED;");
            ctx.write_line(&format!("state->exit_code = {code};"));
            ctx.write_line("return;");
        }
        Terminator::Trap { message } => {
            let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
            ctx.sync_regs_to_state();
            ctx.write_line(&format!("state->pc = {};", hex_u32(pc)));
            ctx.write_line("state->has_exited = OPENVM_EXEC_TRAPPED;");
            ctx.write_line("state->exit_code = 0;");
            ctx.write_line(&format!("/* TRAP: {escaped} */"));
            ctx.write_line("return;");
        }
        Terminator::Extension(ext) => {
            let branch_to = |target: u32| -> String {
                if tc.valid_blocks.contains(&target) {
                    format!("[[clang::musttail]] return block_0x{target:08x}({args});")
                } else {
                    format!(
                        "[[clang::musttail]] return dispatch_table[dispatch_index({})]({args});",
                        hex_u32(target)
                    )
                }
            };
            ext.emit_c_term(ctx, &branch_to);
        }
    }
}

/// Emit a tail call to a known PC. Uses a direct call if the target is a valid
/// block; otherwise falls back to the dispatch table.
fn emit_tail_call(
    ctx: &mut EmitContext,
    target: u32,
    args: &str,
    valid_blocks: &std::collections::HashSet<u32>,
) {
    if valid_blocks.contains(&target) {
        ctx.write_line(&format!(
            "[[clang::musttail]] return block_0x{target:08x}({args});"
        ));
    } else {
        ctx.write_line(&format!(
            "[[clang::musttail]] return dispatch_table[dispatch_index({})]({args});",
            hex_u32(target)
        ));
    }
}

// ── ALU helpers ──────────────────────────────────────────────────────────────

fn alu_expr(op: AluOp, left: &str, right: &str) -> String {
    match op {
        AluOp::Add => format!("{left} + {right}"),
        AluOp::Sub => format!("{left} - {right}"),
        AluOp::Sll => format!("{left} << ({right} & 0x1fu)"),
        AluOp::Slt => format!("((int32_t){left} < (int32_t){right}) ? 1u : 0u"),
        AluOp::Sltu => format!("({left} < {right}) ? 1u : 0u"),
        AluOp::Xor => format!("{left} ^ {right}"),
        AluOp::Srl => format!("{left} >> ({right} & 0x1fu)"),
        AluOp::Sra => format!("(uint32_t)((int32_t){left} >> ({right} & 0x1fu))"),
        AluOp::Or => format!("{left} | {right}"),
        AluOp::And => format!("{left} & {right}"),
    }
}

fn shift_imm_expr(op: AluOp, val: &str, shamt: u8) -> String {
    match op {
        AluOp::Sll => format!("{val} << {}", hex_u32(shamt as u32)),
        AluOp::Srl => format!("{val} >> {}", hex_u32(shamt as u32)),
        AluOp::Sra => format!("(uint32_t)((int32_t){val} >> {})", hex_u32(shamt as u32)),
        _ => unreachable!("invalid shift op {op:?}"),
    }
}

/// Format an i32 immediate as a C uint32_t literal.
fn imm_literal(imm: i32) -> String {
    hex_u32(imm as u32)
}

pub(super) fn hex_u32(value: u32) -> String {
    format!("0x{value:08x}u")
}

fn branch_cond_expr(cond: BranchCond, left: &str, right: &str) -> String {
    match cond {
        BranchCond::Eq => format!("{left} == {right}"),
        BranchCond::Ne => format!("{left} != {right}"),
        BranchCond::Lt => format!("(int32_t){left} < (int32_t){right}"),
        BranchCond::Ge => format!("(int32_t){left} >= (int32_t){right}"),
        BranchCond::Ltu => format!("{left} < {right}"),
        BranchCond::Geu => format!("{left} >= {right}"),
    }
}

// ── MulDiv ──────────────────────────────────────────────────────────────────

fn emit_muldiv(ctx: &mut EmitContext, op: MulDivOp, rd: u8, rs1: u8, rs2: u8) {
    let l = ctx.read_reg(rs1);
    let r = ctx.read_reg(rs2);

    match op {
        MulDivOp::Mul => {
            let expr = format!("(uint32_t)((int64_t)(int32_t){l} * (int64_t)(int32_t){r})");
            ctx.write_reg(rd, &expr);
        }
        MulDivOp::Mulh => {
            let expr = format!("(uint32_t)(((int64_t)(int32_t){l} * (int64_t)(int32_t){r}) >> 32)");
            ctx.write_reg(rd, &expr);
        }
        MulDivOp::Mulhsu => {
            let expr = format!("(uint32_t)(((int64_t)(int32_t){l} * (uint64_t){r}) >> 32)");
            ctx.write_reg(rd, &expr);
        }
        MulDivOp::Mulhu => {
            let expr = format!("(uint32_t)(((uint64_t){l} * (uint64_t){r}) >> 32)");
            ctx.write_reg(rd, &expr);
        }
        MulDivOp::Div => {
            ctx.write_reg(rd, &format!("rv_div((int32_t){l}, (int32_t){r})"));
        }
        MulDivOp::Divu => {
            ctx.write_reg(rd, &format!("rv_divu({l}, {r})"));
        }
        MulDivOp::Rem => {
            ctx.write_reg(rd, &format!("rv_rem((int32_t){l}, (int32_t){r})"));
        }
        MulDivOp::Remu => {
            ctx.write_reg(rd, &format!("rv_remu({l}, {r})"));
        }
    }
}
