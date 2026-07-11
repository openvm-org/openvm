//! C code generation for IR instructions and terminators.
use std::collections::HashSet;

use openvm_instructions::program::DEFAULT_PC_STEP;
use rvr_openvm_ir::{CfgBranchCond, CfgIntWidth, CfgOperand, CfgTerm, ExtEmitCtx, ExtInstr, Terminator};

use super::context::EmitContext;

/// Context for terminator code generation and tail-call dispatch.
pub struct TermCtx<'a> {
    /// Set of valid block start PCs for direct tail calls.
    pub valid_blocks: &'a HashSet<u64>,
}

/// Whether the current compact-record phase migrates this extension-owned
/// instruction. #3059 moved RV64I into `ExtInstr` nodes, so the instruction's
/// stable opcode name replaces the removed monolithic `Instr` enum match.
pub fn instr_emits_inline_record(instr: &dyn ExtInstr) -> bool {
    matches!(
        instr.opname(),
        "add"
            | "sub"
            | "addi"
            | "sll"
            | "slt"
            | "sltu"
            | "xor"
            | "srl"
            | "sra"
            | "or"
            | "and"
            | "slli"
            | "slti"
            | "sltiu"
            | "xori"
            | "srli"
            | "srai"
            | "ori"
            | "andi"
            | "mul"
            | "mulh"
            | "mulhsu"
            | "mulhu"
            | "div"
            | "divu"
            | "rem"
            | "remu"
            | "mulw"
            | "divw"
            | "divuw"
            | "remw"
            | "remuw"
    )
}

/// Emit C code for a terminator using tail calls between blocks.
///
/// Static block-start targets use direct tail calls; other static targets trap.
/// Dynamic targets use the dispatch table. Exit and trap write cached registers
/// back to `RvState` before returning from the generated block.
pub fn emit_terminator(ctx: &mut EmitContext, term: &Terminator, pc: u64, tc: &TermCtx) {
    let next_pc = pc.wrapping_add(u64::from(DEFAULT_PC_STEP));
    let args = ctx.tail_call_args();

    match term.cfg_term(pc, next_pc) {
        CfgTerm::FallThrough => emit_tail_call(ctx, next_pc, &args, tc.valid_blocks),
        CfgTerm::Jump {
            link_dst, target, ..
        } => {
            if let Some(dst) = link_dst {
                ctx.write_var(dst, &hex_u64(next_pc));
            } else {
                ctx.trace_timestamp();
            }
            emit_tail_call(ctx, target, &args, tc.valid_blocks);
        }
        CfgTerm::JumpIndirect {
            link_dst,
            base_value,
            offset,
            target_mask,
            ..
        } => {
            let base_value = match base_value {
                CfgOperand::Var(base) => {
                    let value = ctx.read_var(base);
                    if link_dst.is_some_and(|dst| dst == base) {
                        // Save the base before writing the same variable as the link
                        // destination, preserving the jump target across the write.
                        ctx.materialize_u64(&value)
                    } else {
                        value
                    }
                }
                CfgOperand::Const(value) => hex_u64(value),
            };
            if let Some(dst) = link_dst {
                ctx.write_var(dst, &hex_u64(next_pc));
            } else {
                ctx.trace_timestamp();
            }
            let target = indirect_target_expr(ctx, &base_value, offset, target_mask);
            ctx.write_line(&format!(
                "if (unlikely(!rv_pc_is_dispatchable({target}))) {{"
            ));
            ctx.write_line(&format!("  [[clang::musttail]] return rv_trap({args});"));
            ctx.write_line("}");
            ctx.write_line(&format!("state->pc = {target};"));
            ctx.write_line(&format!(
                "[[clang::musttail]] return dispatch_table[rv_dispatch_index({target})]({args});"
            ));
        }
        CfgTerm::Branch {
            cond,
            width,
            lhs,
            rhs,
            target,
            known,
        } => {
            if let Some(taken) = known {
                if ctx.traces_values() {
                    ctx.read_var(lhs);
                    ctx.read_var(rhs);
                }
                emit_tail_call(
                    ctx,
                    if taken { target } else { next_pc },
                    &args,
                    tc.valid_blocks,
                );
                return;
            }
            let lhs = ctx.read_var(lhs);
            let rhs = ctx.read_var(rhs);
            let cmp = branch_cond_expr(cond, width, &lhs, &rhs);
            ctx.write_line(&format!("if ({cmp}) {{"));
            ctx.write_line(&format!(
                "  {}",
                static_tail_call(target, &args, tc.valid_blocks)
            ));
            ctx.write_line("}");
            emit_tail_call(ctx, next_pc, &args, tc.valid_blocks);
        }
        CfgTerm::Exit { code } => emit_exit(ctx, pc, code),
        CfgTerm::Trap { message } => emit_trap(ctx, pc, &message),
        CfgTerm::Opaque { .. } => {
            let Terminator::Instruction { node, .. } = term else {
                unreachable!("opaque control flow requires an instruction-owned terminator")
            };
            let branch_to = |target| static_tail_call(target, &args, tc.valid_blocks);
            node.emit_c_term(ctx, &branch_to);
        }
    }
}

fn indirect_target_expr(
    ctx: &mut EmitContext,
    base: &str,
    offset: i32,
    target_mask: u64,
) -> String {
    let sum = if offset == 0 {
        base.to_string()
    } else if offset > 0 {
        format!("({base} + {})", hex_u64(offset as u64))
    } else {
        format!("({base} - {})", hex_u64(offset.unsigned_abs() as u64))
    };
    ctx.materialize_u64(&format!("{sum} & {}", hex_u64(target_mask)))
}

fn emit_exit(ctx: &mut EmitContext, pc: u64, code: u32) {
    ctx.sync_regs_to_state();
    ctx.write_line(&format!(
        "rv_set_status_at(state, {}, OPENVM_EXEC_TERMINATED, {code});",
        hex_u64(pc)
    ));
    ctx.write_line("return;");
}

fn emit_trap(ctx: &mut EmitContext, pc: u64, message: &str) {
    let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
    ctx.sync_regs_to_state();
    ctx.write_line(&format!(
        "rv_set_status_at(state, {}, OPENVM_EXEC_TRAPPED, 0);",
        hex_u64(pc)
    ));
    ctx.write_line(&format!("/* TRAP: {escaped} */"));
    ctx.write_line("return;");
}

/// Emit a direct tail call for a block-start PC or a trap for any other PC.
fn static_tail_call(target: u64, args: &str, valid_blocks: &HashSet<u64>) -> String {
    if valid_blocks.contains(&target) {
        format!("[[clang::musttail]] return block_0x{target:08x}({args});")
    } else {
        format!("state->pc = 0x{target:016x}ull; [[clang::musttail]] return rv_trap({args});")
    }
}

fn emit_tail_call(ctx: &mut EmitContext, target: u64, args: &str, valid_blocks: &HashSet<u64>) {
    ctx.write_line(&static_tail_call(target, args, valid_blocks));
}

fn branch_cond_expr(cond: CfgBranchCond, width: CfgIntWidth, left: &str, right: &str) -> String {
    let (left, right) = match width {
        CfgIntWidth::U32 => (format!("(uint32_t){left}"), format!("(uint32_t){right}")),
        CfgIntWidth::U64 => (left.to_string(), right.to_string()),
    };
    match (cond, width) {
        (CfgBranchCond::Eq, _) => format!("{left} == {right}"),
        (CfgBranchCond::Ne, _) => format!("{left} != {right}"),
        (CfgBranchCond::LessThanSigned, CfgIntWidth::U64) => {
            format!("(int64_t){left} < (int64_t){right}")
        }
        (CfgBranchCond::GreaterEqualSigned, CfgIntWidth::U64) => {
            format!("(int64_t){left} >= (int64_t){right}")
        }
        (CfgBranchCond::LessThanUnsigned, CfgIntWidth::U64) => format!("{left} < {right}"),
        (CfgBranchCond::GreaterEqualUnsigned, CfgIntWidth::U64) => format!("{left} >= {right}"),
        (CfgBranchCond::LessThanSigned, CfgIntWidth::U32) => {
            format!("(int32_t){left} < (int32_t){right}")
        }
        (CfgBranchCond::GreaterEqualSigned, CfgIntWidth::U32) => {
            format!("(int32_t){left} >= (int32_t){right}")
        }
        (CfgBranchCond::LessThanUnsigned, CfgIntWidth::U32) => format!("{left} < {right}"),
        (CfgBranchCond::GreaterEqualUnsigned, CfgIntWidth::U32) => format!("{left} >= {right}"),
    }
}

pub(super) fn hex_u64(value: u64) -> String {
    format!("0x{value:016x}ull")
}

pub(super) fn hex_u32(value: u32) -> String {
    format!("0x{value:08x}u")
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{CfgEffect, ExtInstr, Variable};

    use super::*;
    use crate::emit::context::{BlockAbi, EmitMode};

    #[derive(Clone, Debug)]
    struct KnownBranch;

    impl ExtInstr for KnownBranch {
        fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

        fn cfg_effect(&self) -> CfgEffect {
            CfgEffect::None
        }

        fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
            Some(CfgTerm::Branch {
                cond: CfgBranchCond::Eq,
                width: CfgIntWidth::U64,
                lhs: Variable::new(1),
                rhs: Variable::new(2),
                target: 8,
                known: Some(true),
            })
        }

        fn clone_box(&self) -> Box<dyn ExtInstr> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn known_branch_reads_operands_only_for_value_tracing() {
        let term = Terminator::instruction(KnownBranch);
        let valid_blocks = HashSet::from([8]);
        let tc = TermCtx {
            valid_blocks: &valid_blocks,
        };

        let mut direct = EmitContext::new(
            HashSet::new(),
            EmitMode::Direct,
            BlockAbi::Plain,
            None,
            None,
        );
        emit_terminator(&mut direct, &term, 0, &tc);
        assert!(!direct.buf().contains("reg_read"));

        let mut tracing = EmitContext::new(
            HashSet::new(),
            EmitMode::ValueTrace,
            BlockAbi::Plain,
            None,
            None,
        );
        emit_terminator(&mut tracing, &term, 0, &tc);
        assert_eq!(tracing.buf().matches("trace_reg_read").count(), 2);
    }

    #[test]
    fn branch_width_controls_operand_truncation() {
        assert_eq!(
            branch_cond_expr(CfgBranchCond::Eq, CfgIntWidth::U64, "left", "right"),
            "left == right"
        );
        assert_eq!(
            branch_cond_expr(CfgBranchCond::Eq, CfgIntWidth::U32, "left", "right"),
            "(uint32_t)left == (uint32_t)right"
        );
        assert_eq!(
            branch_cond_expr(
                CfgBranchCond::LessThanSigned,
                CfgIntWidth::U32,
                "left",
                "right",
            ),
            "(int32_t)(uint32_t)left < (int32_t)(uint32_t)right"
        );
        assert_eq!(
            branch_cond_expr(
                CfgBranchCond::GreaterEqualUnsigned,
                CfgIntWidth::U32,
                "left",
                "right",
            ),
            "(uint32_t)left >= (uint32_t)right"
        );
    }
}
