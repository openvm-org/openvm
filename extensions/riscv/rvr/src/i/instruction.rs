//! RV64I instruction nodes and C code generation.

use rvr_openvm_ir::{
    CfgBranchCond, CfgEffect, CfgIntWidth, CfgJumpKind, CfgOp, CfgOperand, CfgResultWidth, CfgTerm,
    ExtEmitCtx, ExtInstr, MemWidth,
};

use crate::instruction::{hex_u64, reg_operand, Reg, RA, SP, ZERO};

/// RV64I arithmetic or logical operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
}

impl AluOp {
    fn cfg_op(self) -> CfgOp {
        match self {
            Self::Add => CfgOp::Add,
            Self::Sub => CfgOp::Sub,
            Self::Sll => CfgOp::ShiftLeft,
            Self::Slt => CfgOp::LessThanSigned,
            Self::Sltu => CfgOp::LessThanUnsigned,
            Self::Xor => CfgOp::Xor,
            Self::Srl => CfgOp::ShiftRightLogical,
            Self::Sra => CfgOp::ShiftRightArithmetic,
            Self::Or => CfgOp::Or,
            Self::And => CfgOp::And,
        }
    }

    fn name(self, immediate: bool, word: bool) -> &'static str {
        match (self, immediate, word) {
            (Self::Add, false, false) => "add",
            (Self::Sub, false, false) => "sub",
            (Self::Sll, false, false) => "sll",
            (Self::Slt, false, false) => "slt",
            (Self::Sltu, false, false) => "sltu",
            (Self::Xor, false, false) => "xor",
            (Self::Srl, false, false) => "srl",
            (Self::Sra, false, false) => "sra",
            (Self::Or, false, false) => "or",
            (Self::And, false, false) => "and",
            (Self::Add, true, false) => "addi",
            (Self::Sll, true, false) => "slli",
            (Self::Slt, true, false) => "slti",
            (Self::Sltu, true, false) => "sltiu",
            (Self::Xor, true, false) => "xori",
            (Self::Srl, true, false) => "srli",
            (Self::Sra, true, false) => "srai",
            (Self::Or, true, false) => "ori",
            (Self::And, true, false) => "andi",
            (Self::Add, false, true) => "addw",
            (Self::Sub, false, true) => "subw",
            (Self::Sll, false, true) => "sllw",
            (Self::Srl, false, true) => "srlw",
            (Self::Sra, false, true) => "sraw",
            (Self::Add, true, true) => "addiw",
            (Self::Sll, true, true) => "slliw",
            (Self::Srl, true, true) => "srliw",
            (Self::Sra, true, true) => "sraiw",
            _ => "alu",
        }
    }
}

/// An RV64I instruction implemented by this extension.
#[derive(Debug, Clone)]
pub(crate) enum RiscvIInstr {
    /// Register-register or register-immediate arithmetic.
    Alu {
        op: AluOp,
        word: bool,
        immediate: bool,
        rd: Reg,
        lhs: Reg,
        rhs: CfgOperand,
        /// Original register operand before CFG constant folding.
        rhs_reg: Option<Reg>,
    },
    /// Load from main memory.
    Load {
        width: MemWidth,
        signed: bool,
        rd: Reg,
        base: Reg,
        offset: i16,
    },
    /// Store to main memory.
    Store {
        width: MemWidth,
        base: Reg,
        src: Reg,
        offset: i16,
    },
    /// Write a precomputed LUI or AUIPC result.
    Const {
        name: &'static str,
        rd: Reg,
        value: u64,
    },
    /// Conditional branch.
    Branch {
        cond: CfgBranchCond,
        lhs: Reg,
        rhs: Reg,
        target: u64,
    },
    /// Jump to a statically known target.
    Jump { link_dst: Option<Reg>, target: u64 },
    /// Jump through a register to one of the targets found by CFG analysis.
    JumpIndirect {
        link_dst: Option<Reg>,
        base: Reg,
        offset: i32,
    },
}

impl ExtInstr for RiscvIInstr {
    fn opname(&self) -> &str {
        match self {
            Self::Alu {
                op,
                word,
                immediate,
                ..
            } => op.name(*immediate, *word),
            Self::Load { width, signed, .. } => match (width, signed) {
                (MemWidth::Double, _) => "ld",
                (MemWidth::Word, true) => "lw",
                (MemWidth::Word, false) => "lwu",
                (MemWidth::Half, true) => "lh",
                (MemWidth::Half, false) => "lhu",
                (MemWidth::Byte, true) => "lb",
                (MemWidth::Byte, false) => "lbu",
            },
            Self::Store { width, .. } => match width {
                MemWidth::Double => "sd",
                MemWidth::Word => "sw",
                MemWidth::Half => "sh",
                MemWidth::Byte => "sb",
            },
            Self::Const { name, .. } => name,
            Self::Branch { cond, .. } => match cond {
                CfgBranchCond::Eq => "beq",
                CfgBranchCond::Ne => "bne",
                CfgBranchCond::LessThanSigned => "blt",
                CfgBranchCond::GreaterEqualSigned => "bge",
                CfgBranchCond::LessThanUnsigned => "bltu",
                CfgBranchCond::GreaterEqualUnsigned => "bgeu",
            },
            Self::Jump {
                link_dst: Some(_), ..
            } => "jal",
            Self::Jump { link_dst: None, .. } => "j",
            Self::JumpIndirect {
                link_dst: Some(_), ..
            } => "jalr",
            Self::JumpIndirect { link_dst: None, .. } => "jr",
        }
    }

    fn accesses_memory(&self) -> bool {
        matches!(self, Self::Load { .. } | Self::Store { .. })
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        match self {
            Self::Branch { .. } | Self::Jump { .. } | Self::JumpIndirect { .. } => {}
            Self::Alu {
                op,
                word,
                rd,
                lhs,
                rhs,
                rhs_reg,
                ..
            } => {
                let lhs_value = ctx.read_var(*lhs);
                let rhs_value = rhs_reg
                    .map(|reg| ctx.read_var(reg))
                    .unwrap_or_else(|| operand_c(ctx, *rhs));
                let value = (!*word)
                    .then(|| constant_alu_result(*op, *lhs, *rhs))
                    .flatten()
                    .map(hex_u64)
                    .unwrap_or_else(|| alu_expr(*op, *word, &lhs_value, &rhs_value));
                ctx.write_var(*rd, &value);
            }
            Self::Load {
                width,
                signed,
                rd,
                base,
                offset,
            } => {
                let is_sp_relative = *base == SP;
                let base = ctx.read_var(*base);
                let value = if is_sp_relative {
                    ctx.read_sp_mem(&base, *offset, width.bytes(), *signed)
                } else {
                    ctx.read_mem(&base, *offset, width.bytes(), *signed)
                };
                if *rd == ZERO {
                    // Pure execution still has to perform the potentially
                    // trapping load, but has no register write that would use
                    // the temporary. Preflight's write_var call below reserves
                    // the disabled destination slot.
                    ctx.write_line(&format!("(void){value};"));
                } else {
                    ctx.append_replay_value(&value);
                }
                ctx.write_var(*rd, &value);
            }
            Self::Store {
                width,
                base,
                src,
                offset,
            } => {
                let is_sp_relative = *base == SP;
                let base = ctx.read_var(*base);
                let value = ctx.read_var(*src);
                if is_sp_relative {
                    ctx.write_sp_mem(&base, *offset, &value, width.bytes());
                } else {
                    ctx.write_mem(&base, *offset, &value, width.bytes());
                }
            }
            Self::Const { rd, value, .. } => ctx.write_var(*rd, &hex_u64(*value)),
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn supports_preflight(&self) -> bool {
        true
    }

    fn cfg_effect(&self) -> CfgEffect {
        match self {
            Self::Store { .. }
            | Self::Branch { .. }
            | Self::Jump { .. }
            | Self::JumpIndirect { .. } => CfgEffect::None,
            Self::Alu { rd, .. } | Self::Const { rd, .. } if *rd == ZERO => CfgEffect::None,
            Self::Alu {
                op,
                word,
                rd,
                lhs,
                rhs,
                ..
            } => CfgEffect::WriteOp {
                dst: *rd,
                op: op.cfg_op(),
                lhs: reg_operand(*lhs),
                rhs: *rhs,
                result: if *word {
                    CfgResultWidth::SignExtend32
                } else {
                    CfgResultWidth::U64
                },
            },
            Self::Load { rd, .. } if *rd != ZERO => CfgEffect::WriteUnknown { dst: *rd },
            Self::Load { .. } => CfgEffect::None,
            Self::Const { rd, value, .. } => CfgEffect::WriteConst {
                dst: *rd,
                value: *value,
            },
        }
    }

    fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
        match self {
            Self::Branch {
                cond,
                lhs,
                rhs,
                target,
            } => Some(CfgTerm::Branch {
                cond: *cond,
                width: CfgIntWidth::U64,
                lhs: *lhs,
                rhs: *rhs,
                target: *target,
                known: known_branch_result(*cond, *lhs, *rhs),
            }),
            Self::Jump { link_dst, target } => Some(CfgTerm::Jump {
                kind: if link_dst.is_some() {
                    CfgJumpKind::Call
                } else {
                    CfgJumpKind::Jump
                },
                link_dst: *link_dst,
                has_link_write_slot: true,
                target: *target,
            }),
            Self::JumpIndirect {
                link_dst,
                base,
                offset,
            } => Some(CfgTerm::JumpIndirect {
                kind: if link_dst.is_some() {
                    CfgJumpKind::Call
                } else if *base == RA {
                    CfgJumpKind::Return
                } else {
                    CfgJumpKind::Jump
                },
                link_dst: *link_dst,
                has_link_write_slot: true,
                base_value: if *base == ZERO {
                    CfgOperand::ReadConst {
                        source: *base,
                        value: 0,
                    }
                } else {
                    CfgOperand::Var(*base)
                },
                offset: *offset,
                target_mask: !1,
            }),
            _ => None,
        }
    }
}

fn operand_c(ctx: &mut dyn ExtEmitCtx, operand: CfgOperand) -> String {
    match operand {
        CfgOperand::Var(reg) => ctx.read_var(reg),
        CfgOperand::Const(value) => hex_u64(value),
        CfgOperand::ReadConst { source, value } => {
            ctx.read_var(source);
            hex_u64(value)
        }
    }
}

fn alu_expr(op: AluOp, word: bool, lhs: &str, rhs: &str) -> String {
    if word {
        let inner = match op {
            AluOp::Add => format!("(uint32_t){lhs} + (uint32_t){rhs}"),
            AluOp::Sub => format!("(uint32_t){lhs} - (uint32_t){rhs}"),
            AluOp::Sll => format!("(uint32_t){lhs} << ((uint32_t){rhs} & 0x1fu)"),
            AluOp::Srl => format!("(uint32_t){lhs} >> ((uint32_t){rhs} & 0x1fu)"),
            AluOp::Sra => {
                format!("(uint32_t)((int32_t)(uint32_t){lhs} >> ((uint32_t){rhs} & 0x1fu))")
            }
            _ => unreachable!("invalid RV64 W operation"),
        };
        return format!("(uint64_t)(int32_t)({inner})");
    }

    match op {
        AluOp::Add => format!("{lhs} + {rhs}"),
        AluOp::Sub => format!("{lhs} - {rhs}"),
        AluOp::Sll => format!("{lhs} << ({rhs} & 0x3fu)"),
        AluOp::Slt => format!("(int64_t){lhs} < (int64_t){rhs}"),
        AluOp::Sltu => format!("{lhs} < {rhs}"),
        AluOp::Xor => format!("{lhs} ^ {rhs}"),
        AluOp::Srl => format!("{lhs} >> ({rhs} & 0x3fu)"),
        AluOp::Sra => format!("(uint64_t)((int64_t){lhs} >> ({rhs} & 0x3fu))"),
        AluOp::Or => format!("{lhs} | {rhs}"),
        AluOp::And => format!("{lhs} & {rhs}"),
    }
}

fn constant_alu_result(op: AluOp, lhs: Reg, rhs: CfgOperand) -> Option<u64> {
    match (op, rhs) {
        (AluOp::Slt | AluOp::Sltu, CfgOperand::Var(rhs)) if lhs == rhs => Some(0),
        (AluOp::Sltu, CfgOperand::Const(0)) => Some(0),
        (AluOp::Slt, CfgOperand::Const(rhs)) if lhs == ZERO => Some(u64::from(0 < rhs as i64)),
        (AluOp::Sltu, CfgOperand::Const(rhs)) if lhs == ZERO => Some(u64::from(rhs != 0)),
        _ => None,
    }
}

fn known_branch_result(cond: CfgBranchCond, lhs: Reg, rhs: Reg) -> Option<bool> {
    if lhs == rhs {
        return Some(matches!(
            cond,
            CfgBranchCond::Eq
                | CfgBranchCond::GreaterEqualSigned
                | CfgBranchCond::GreaterEqualUnsigned
        ));
    }
    match cond {
        CfgBranchCond::LessThanUnsigned if rhs == ZERO => Some(false),
        CfgBranchCond::GreaterEqualUnsigned if rhs == ZERO => Some(true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::PageAddressSpace;

    use super::*;

    #[derive(Default)]
    struct RecordingCtx {
        operations: Vec<String>,
    }

    impl ExtEmitCtx for RecordingCtx {
        fn read_var(&mut self, var: Reg) -> String {
            format!("r{}", var.index())
        }

        fn peek_var(&mut self, var: Reg) -> String {
            self.read_var(var)
        }

        fn advance_timestamp(&mut self, _slots: u32) {}

        fn write_var(&mut self, var: Reg, val: &str) {
            self.operations
                .push(format!("write:r{}={val}", var.index()));
        }

        fn write_line(&mut self, line: &str) {
            self.operations.push(format!("line:{line}"));
        }

        fn emit_trap(&mut self) {
            unreachable!()
        }

        fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
            self.operations
                .push(format!("read:{base}:{offset}:{width}:{signed}"));
            "loaded".to_string()
        }

        fn read_sp_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
            self.operations
                .push(format!("read-sp:{base}:{offset}:{width}:{signed}"));
            "loaded".to_string()
        }

        fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
            self.operations
                .push(format!("write:{base}:{offset}:{val}:{width}"));
        }

        fn write_sp_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
            self.operations
                .push(format!("write-sp:{base}:{offset}:{val}:{width}"));
        }

        fn write_aligned_mem_block(&mut self, _addr: &str, _val: &str) {
            unreachable!()
        }

        fn reserve_preflight_timestamp_slots(&mut self, _slots: &str) {}

        fn append_replay_value(&mut self, value: &str) {
            self.operations.push(format!("append:{value}"));
        }

        fn emit_call(&mut self, _name: &str, _args: &[&str]) {
            unreachable!()
        }

        fn emit_call_without_page_flush(&mut self, _name: &str, _args: &[&str]) {
            unreachable!()
        }

        fn emit_call_expr(&mut self, _ret_ty: &str, _name: &str, _args: &[&str]) -> String {
            unreachable!()
        }

        fn emit_call_with_trace_result(
            &mut self,
            _ret_ty: &str,
            _name: &str,
            _args: &[&str],
        ) -> Option<String> {
            unreachable!()
        }

        fn trace_chip(&mut self, _chip_idx: u32, _count_expr: &str) {}

        fn trace_chip_if_nonzero(&mut self, _chip_idx: u32, _count_expr: &str) {}

        fn trace_page_access(
            &mut self,
            _addr: &str,
            _width: MemWidth,
            _addr_space: PageAddressSpace,
        ) {
        }

        fn trace_page_access_u64_range(
            &mut self,
            _base_addr: &str,
            _num_dwords: &str,
            _addr_space: PageAddressSpace,
        ) {
        }
    }

    fn load(rd: Reg) -> RiscvIInstr {
        RiscvIInstr::Load {
            width: MemWidth::Word,
            signed: true,
            rd,
            base: Reg::new(2),
            offset: 4,
        }
    }

    #[test]
    fn load_appends_only_architectural_destination_results() {
        let mut enabled = RecordingCtx::default();
        load(Reg::new(3)).emit_c(&mut enabled);
        assert_eq!(
            enabled.operations,
            ["read-sp:r2:4:4:true", "append:loaded", "write:r3=loaded"]
        );

        let mut x0 = RecordingCtx::default();
        load(ZERO).emit_c(&mut x0);
        assert_eq!(
            x0.operations,
            [
                "read-sp:r2:4:4:true",
                "line:(void)loaded;",
                "write:r0=loaded"
            ]
        );
    }

    #[test]
    fn memory_accesses_select_the_sp_cache_only_for_x2() {
        let mut ctx = RecordingCtx::default();
        RiscvIInstr::Load {
            width: MemWidth::Byte,
            signed: false,
            rd: Reg::new(3),
            base: Reg::new(4),
            offset: -1,
        }
        .emit_c(&mut ctx);
        RiscvIInstr::Store {
            width: MemWidth::Double,
            base: SP,
            src: Reg::new(5),
            offset: 8,
        }
        .emit_c(&mut ctx);
        RiscvIInstr::Store {
            width: MemWidth::Word,
            base: Reg::new(6),
            src: Reg::new(7),
            offset: 12,
        }
        .emit_c(&mut ctx);

        assert_eq!(
            ctx.operations,
            [
                "read:r4:-1:1:false",
                "append:loaded",
                "write:r3=loaded",
                "write-sp:r2:8:r5:8",
                "write:r6:12:r7:4",
            ]
        );
    }
}
