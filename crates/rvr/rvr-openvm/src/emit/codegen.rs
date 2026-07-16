//! C code generation for IR instructions and terminators.
use std::collections::HashSet;

use openvm_instructions::program::DEFAULT_PC_STEP;
use rvr_openvm_ir::{
    ArenaRw1Baked, ArenaWr1Baked, CfgBranchCond, CfgIntWidth, CfgOperand, CfgTerm, ExtEmitCtx,
    ExtInstr, InlineRecordShape, Terminator, Variable,
};

use super::context::{ArenaBranch2Baked, EmitContext};

/// Context for terminator code generation and tail-call dispatch.
pub struct TermCtx<'a> {
    /// Set of valid block start PCs for direct tail calls.
    pub valid_blocks: &'a HashSet<u64>,
}

/// R4 arena-native geometry for one air's full (adapter + core) record.
/// Values are computed by the host from the real record types and baked into
/// the generated per-air emitter, so the C side never mirrors Rust structs.
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

/// Per-shape field offsets. Adapter offsets are relative to the record start;
/// core offsets are relative to the core record start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaNativeLayout {
    AddI(AddIArenaFieldOffsets),
    Alu3(Alu3ArenaFieldOffsets),
    Branch2(Branch2ArenaFieldOffsets),
    LoadStore(LoadStoreArenaFieldOffsets),
    Wr1(Wr1ArenaFieldOffsets),
    Rw1(Rw1ArenaFieldOffsets),
    /// The extension emits the complete record through its own C shim. When
    /// `residual_memory_chronology` is true, all timestamp-bearing accesses
    /// remain in the memory log, so delta mode needs no duplicate program-log
    /// entry for chronology replay. `layout_id` is an explicit, versioned
    /// extension-owned identity; changing the final record layout requires a
    /// new value and therefore a new G2 schema fingerprint.
    Custom {
        residual_memory_chronology: bool,
        layout_id: &'static str,
    },
    /// Extension-owned packed records with runtime row counts.
    CustomVariableRows {
        residual_memory_chronology: bool,
    },
}

/// Offsets for the AddI full record: a one-read one-u16-block-write
/// immediate adapter and the dedicated AddI core witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddIArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rd_ptr: usize,
    pub rs1_ptr: usize,
    pub read_prev_ts: usize,
    pub write_prev_ts: usize,
    pub write_prev_data: usize,
    pub core_rs1: usize,
    pub core_imm_low11: usize,
    pub core_imm_sign: usize,
}

/// Field offsets for a conditional one-write JAL/LUI/AUIPC record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wr1ArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rd_ptr: usize,
    pub rd_prev_ts: usize,
    pub rd_prev_data: usize,
    pub core_imm: usize,
    pub core_rd_data: usize,
    pub core_is_jal: usize,
    pub core_from_pc: usize,
}

/// Field offsets for a read/conditional-write JALR record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rw1ArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rs1_ptr: usize,
    pub rd_ptr: usize,
    pub read_prev_ts: usize,
    pub write_prev_ts: usize,
    pub write_prev_data: usize,
    pub core_imm: usize,
    pub core_from_pc: usize,
    pub core_rs1_val: usize,
    pub core_imm_sign: usize,
}

/// Offsets for a load or store full record. Adapter/core fields that are not
/// present in a width-specific record use the `usize::MAX` sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadStoreArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rs1_ptr: usize,
    pub rs1_val: usize,
    pub rs1_aux_prev_ts: usize,
    pub rd_rs2_ptr: usize,
    pub read_data_aux_prev_ts: usize,
    /// Second block read timestamp for multi-byte accesses, or `usize::MAX`.
    pub read_data_aux_prev_ts2: usize,
    /// Stored as a u16.
    pub imm: usize,
    pub imm_sign: usize,
    pub mem_as: usize,
    pub write_prev_ts: usize,
    /// Second previous-write timestamp for multi-byte stores, or `usize::MAX`.
    pub write_prev_ts2: usize,
    /// Adapter-relative previous-write data, or `usize::MAX` when the
    /// previous data lives in the core record.
    pub write_prev_data: usize,
    pub core_local_opcode: usize,
    pub core_is_byte: usize,
    pub core_is_word: usize,
    pub core_shift_amount: usize,
    pub core_read_data: usize,
    /// Second block read data for multi-byte loads, or `usize::MAX`.
    pub core_read_data2: usize,
    pub core_prev_data: usize,
    /// Second previous block for multi-byte stores, or `usize::MAX`.
    pub core_prev_data2: usize,
}

/// BabyBear modulus used to field-canonicalize negative branch offsets.
pub const BABYBEAR_ORDER_U32: u32 = 0x7800_0001;

/// Field offsets for a two-read, no-write branch record.
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

/// Field offsets for a two-read, one-u16-block-write ALU record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alu3ArenaFieldOffsets {
    pub from_pc: usize,
    pub from_timestamp: usize,
    pub rd_ptr: usize,
    pub rs1_ptr: usize,
    pub rs2: usize,
    /// `usize::MAX` when the adapter always reads rs2 as a register.
    pub rs2_as: usize,
    /// `usize::MAX` for byte adapters without an immediate-sign field.
    pub rs2_imm_sign: usize,
    pub reads_aux0_prev_ts: usize,
    pub reads_aux1_prev_ts: usize,
    pub write_prev_ts: usize,
    pub write_prev_data: usize,
    pub core_b: usize,
    pub core_c: usize,
    /// `usize::MAX` for single-opcode cores without this field.
    pub core_local_opcode: usize,
    /// Extra fields present in RV64 W adapters.
    pub w: Option<Alu3WArenaFieldOffsets>,
}

/// W-specific adapter offsets for arena-native ALU records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alu3WArenaFieldOffsets {
    pub rs1_high: usize,
    pub rs2_high: usize,
    pub result_word_msl: usize,
    pub result_sign: usize,
    pub result_word_msl_shift: u8,
    pub result_word_msl_bytes: u8,
}

pub fn inline_record_shape_for_instr(instr: &dyn ExtInstr) -> Option<InlineRecordShape> {
    instr.inline_record_shape()
}

pub fn inline_record_shape_for_terminator(term: &Terminator) -> Option<InlineRecordShape> {
    match term {
        Terminator::Instruction { node, .. } => node.inline_record_shape(),
        _ => None,
    }
}

/// Whether the compact-record transport migrates this instruction.
pub fn instr_emits_inline_record(instr: &dyn ExtInstr) -> bool {
    inline_record_shape_for_instr(instr).is_some()
}

/// Emit C code for a terminator using tail calls between blocks.
///
/// Static block-start targets use direct tail calls; other static targets trap.
/// Dynamic targets use the dispatch table. Exit and trap write cached registers
/// back to `RvState` before returning from the generated block.
pub fn emit_terminator(ctx: &mut EmitContext, term: &Terminator, pc: u64, tc: &TermCtx) {
    let next_pc = pc.wrapping_add(u64::from(DEFAULT_PC_STEP));
    let args = ctx.tail_call_args();
    let inline_shape = ctx
        .inline_records_enabled()
        .then(|| inline_record_shape_for_terminator(term))
        .flatten();

    match term.cfg_term(pc, next_pc) {
        CfgTerm::FallThrough => emit_tail_call(ctx, next_pc, &args, tc.valid_blocks),
        CfgTerm::Jump {
            link_dst, target, ..
        } => {
            if inline_shape == Some(InlineRecordShape::Wr1) {
                let imm = target as i64 - pc as i64;
                let core_imm = if imm >= 0 {
                    imm as u32
                } else {
                    (i64::from(BABYBEAR_ORDER_U32) + imm) as u32
                };
                let baked = ArenaWr1Baked {
                    rd_ptr: link_dst.map_or(u32::MAX, |rd| rd.index() * 8),
                    core_imm,
                    rd_data: next_pc,
                    is_jal: 1,
                    core_from_pc: 0,
                };
                ctx.emit_wr1_inline(
                    link_dst.map(variable_index),
                    &hex_u64(next_pc),
                    Some(baked),
                );
            } else if let Some(dst) = link_dst {
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
            let base_value = if inline_shape == Some(InlineRecordShape::Rw1) {
                let base = match base_value {
                    CfgOperand::Var(base) => variable_index(base),
                    CfgOperand::Const(0) => 0,
                    CfgOperand::Const(_) => {
                        unreachable!("RV64 JALR compact base must be a register or zero")
                    }
                };
                let baked = ArenaRw1Baked {
                    rs1_ptr: u32::from(base) * 8,
                    rd_ptr: link_dst.map_or(u32::MAX, |rd| rd.index() * 8),
                    core_imm: offset as u16,
                    core_imm_sign: (offset < 0) as u8,
                };
                ctx.emit_rw1_inline(
                    link_dst.map(variable_index),
                    base,
                    &hex_u64(next_pc),
                    Some(baked),
                )
            } else {
                match base_value {
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
                }
            };
            if inline_shape != Some(InlineRecordShape::Rw1) {
                if let Some(dst) = link_dst {
                    ctx.write_var(dst, &hex_u64(next_pc));
                } else {
                    ctx.trace_timestamp();
                }
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
            let branch_baked = || {
                let imm = target as i64 - pc as i64;
                let imm = if imm >= 0 {
                    imm as u32
                } else {
                    (i64::from(BABYBEAR_ORDER_U32) + imm) as u32
                };
                let local_opcode = match cond {
                    CfgBranchCond::Eq => 0,
                    CfgBranchCond::Ne => 1,
                    CfgBranchCond::LessThanSigned => 0,
                    CfgBranchCond::LessThanUnsigned => 1,
                    CfgBranchCond::GreaterEqualSigned => 2,
                    CfgBranchCond::GreaterEqualUnsigned => 3,
                };
                ArenaBranch2Baked {
                    rs1_ptr: lhs.index() * 8,
                    rs2_ptr: rhs.index() * 8,
                    imm,
                    local_opcode,
                }
            };
            if let Some(taken) = known {
                if inline_shape == Some(InlineRecordShape::Branch2) {
                    ctx.emit_branch2_inline(
                        variable_index(lhs),
                        variable_index(rhs),
                        Some(branch_baked()),
                    );
                } else if ctx.traces_values() {
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
            let (lhs, rhs) = if inline_shape == Some(InlineRecordShape::Branch2) {
                ctx.emit_branch2_inline(
                    variable_index(lhs),
                    variable_index(rhs),
                    Some(branch_baked()),
                )
            } else {
                (ctx.read_var(lhs), ctx.read_var(rhs))
            };
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

fn variable_index(var: Variable) -> u8 {
    u8::try_from(var.index()).expect("RV64 register index must fit in u8")
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
