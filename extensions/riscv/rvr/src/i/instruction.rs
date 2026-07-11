//! RV64I instruction nodes and C code generation.

use rvr_openvm_ir::{
    CfgBranchCond, CfgEffect, CfgIntWidth, CfgJumpKind, CfgOp, CfgOperand, CfgResultWidth, CfgTerm,
    ExtEmitCtx, ExtInstr, MemWidth,
};

use crate::instruction::{hex_u64, reg_operand, Reg, RA, ZERO};

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
pub(crate) enum Rv64IInstr {
    /// Register-register or register-immediate arithmetic.
    Alu {
        op: AluOp,
        word: bool,
        immediate: bool,
        rd: Reg,
        lhs: Reg,
        rhs: CfgOperand,
    },
    /// Load from main memory.
    Load {
        width: MemWidth,
        signed: bool,
        rd: Option<Reg>,
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

impl ExtInstr for Rv64IInstr {
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
                immediate,
                rd,
                lhs,
                rhs,
            } => {
                if !*word {
                    let result_template = alu_expr(
                        *op,
                        false,
                        "__RVR_LHS__",
                        "__RVR_RHS__",
                    );
                    let emitted = match rhs {
                        CfgOperand::Var(rhs) => {
                            ctx.emit_reg3_inline(*rd, *lhs, *rhs, &result_template)
                        }
                        CfgOperand::Const(imm) if *immediate => ctx.emit_reg2imm_inline(
                            *rd,
                            *lhs,
                            *imm,
                            &result_template,
                        ),
                        CfgOperand::Const(_) => false,
                    };
                    if emitted {
                        return;
                    }
                }
                let lhs_value = ctx.read_var(*lhs);
                let rhs_value = operand_c(ctx, *rhs);
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
                let base = ctx.read_var(*base);
                let value = ctx.read_mem(&base, *offset, width.bytes(), *signed);
                if let Some(rd) = rd {
                    ctx.write_var(*rd, &value);
                } else {
                    ctx.write_line(&format!("(void){value};"));
                }
            }
            Self::Store {
                width,
                base,
                src,
                offset,
            } => {
                let base = ctx.read_var(*base);
                let value = ctx.read_var(*src);
                ctx.write_mem(&base, *offset, &value, width.bytes());
            }
            Self::Const { rd, value, .. } => ctx.write_var(*rd, &hex_u64(*value)),
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        match self {
            Self::Store { .. }
            | Self::Branch { .. }
            | Self::Jump { .. }
            | Self::JumpIndirect { .. } => CfgEffect::None,
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
            Self::Load { rd: Some(rd), .. } => CfgEffect::WriteUnknown { dst: *rd },
            Self::Load { rd: None, .. } => CfgEffect::None,
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
                base_value: reg_operand(*base),
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
