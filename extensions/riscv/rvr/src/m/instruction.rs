//! RV64M instruction nodes and C code generation.

use rvr_openvm_ir::{
    CfgEffect, CfgOp, CfgResultWidth, ExtEmitCtx, ExtInstr, InlineRecordShape,
};

use crate::instruction::{reg_operand, Reg};

/// RV64M multiplication or division operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MulDivOp {
    Mul,
    MulHighSigned,
    MulHighSignedUnsigned,
    MulHighUnsigned,
    DivSigned,
    DivUnsigned,
    RemSigned,
    RemUnsigned,
}

impl MulDivOp {
    fn cfg_op(self) -> CfgOp {
        match self {
            Self::Mul => CfgOp::Mul,
            Self::MulHighSigned => CfgOp::MulHighSigned,
            Self::MulHighSignedUnsigned => CfgOp::MulHighSignedUnsigned,
            Self::MulHighUnsigned => CfgOp::MulHighUnsigned,
            Self::DivSigned => CfgOp::DivSigned,
            Self::DivUnsigned => CfgOp::DivUnsigned,
            Self::RemSigned => CfgOp::RemSigned,
            Self::RemUnsigned => CfgOp::RemUnsigned,
        }
    }

    fn name(self, word: bool) -> &'static str {
        match (self, word) {
            (Self::Mul, false) => "mul",
            (Self::MulHighSigned, false) => "mulh",
            (Self::MulHighSignedUnsigned, false) => "mulhsu",
            (Self::MulHighUnsigned, false) => "mulhu",
            (Self::DivSigned, false) => "div",
            (Self::DivUnsigned, false) => "divu",
            (Self::RemSigned, false) => "rem",
            (Self::RemUnsigned, false) => "remu",
            (Self::Mul, true) => "mulw",
            (Self::DivSigned, true) => "divw",
            (Self::DivUnsigned, true) => "divuw",
            (Self::RemSigned, true) => "remw",
            (Self::RemUnsigned, true) => "remuw",
            _ => "muldiv",
        }
    }
}

/// An RV64M instruction implemented by this extension.
#[derive(Debug, Clone)]
pub(crate) struct Rv64MInstr {
    pub op: MulDivOp,
    pub word: bool,
    pub rd: Reg,
    pub lhs: Reg,
    pub rhs: Reg,
}

impl ExtInstr for Rv64MInstr {
    fn opname(&self) -> &str {
        self.op.name(self.word)
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Alu3)
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let result_template = muldiv_expr(
            self.op,
            self.word,
            "__RVR_LHS__",
            "__RVR_RHS__",
        );
        if ctx.emit_reg3_inline(self.rd, self.lhs, self.rhs, &result_template) {
            return;
        }
        let lhs = ctx.read_var(self.lhs);
        let rhs = ctx.read_var(self.rhs);
        ctx.write_var(self.rd, &muldiv_expr(self.op, self.word, &lhs, &rhs));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::WriteOp {
            dst: self.rd,
            op: self.op.cfg_op(),
            lhs: reg_operand(self.lhs),
            rhs: reg_operand(self.rhs),
            result: if self.word {
                CfgResultWidth::SignExtend32
            } else {
                CfgResultWidth::U64
            },
        }
    }
}

fn muldiv_expr(op: MulDivOp, word: bool, lhs: &str, rhs: &str) -> String {
    if word {
        return match op {
            MulDivOp::Mul => {
                format!("(uint64_t)(int32_t)((uint32_t){lhs} * (uint32_t){rhs})")
            }
            MulDivOp::DivSigned => {
                format!("rv_divw((int32_t)(uint32_t){lhs}, (int32_t)(uint32_t){rhs})")
            }
            MulDivOp::DivUnsigned => format!("rv_divuw((uint32_t){lhs}, (uint32_t){rhs})"),
            MulDivOp::RemSigned => {
                format!("rv_remw((int32_t)(uint32_t){lhs}, (int32_t)(uint32_t){rhs})")
            }
            MulDivOp::RemUnsigned => format!("rv_remuw((uint32_t){lhs}, (uint32_t){rhs})"),
            _ => unreachable!("invalid RV64M W operation"),
        };
    }

    match op {
        MulDivOp::Mul => format!("{lhs} * {rhs}"),
        MulDivOp::MulHighSigned => format!("rv_mulh((int64_t){lhs}, (int64_t){rhs})"),
        MulDivOp::MulHighSignedUnsigned => format!("rv_mulhsu((int64_t){lhs}, {rhs})"),
        MulDivOp::MulHighUnsigned => format!("rv_mulhu({lhs}, {rhs})"),
        MulDivOp::DivSigned => format!("rv_div((int64_t){lhs}, (int64_t){rhs})"),
        MulDivOp::DivUnsigned => format!("rv_divu({lhs}, {rhs})"),
        MulDivOp::RemSigned => format!("rv_rem((int64_t){lhs}, (int64_t){rhs})"),
        MulDivOp::RemUnsigned => format!("rv_remu({lhs}, {rhs})"),
    }
}
