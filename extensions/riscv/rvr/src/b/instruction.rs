//! RV64B instruction nodes and C code generation.

use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr};

use crate::instruction::Reg;

/// RV64B bit-manipulation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitManipOp {
    Sh1Add,
    Sh2Add,
    Sh3Add,
    AddUw,
    Sh1AddUw,
    Sh2AddUw,
    Sh3AddUw,
    SlliUw,
    AndN,
    OrN,
    Xnor,
    Rol,
    Ror,
    Rori,
    RolW,
    RorW,
    RorIw,
    Clz,
    Ctz,
    ClzW,
    CtzW,
    Cpop,
    CpopW,
    Min,
    MinU,
    Max,
    MaxU,
    SextB,
    SextH,
    ZextH,
    OrcB,
    Rev8,
    Bclr,
    Bset,
    Binv,
    Bext,
    BclrI,
    BsetI,
    BinvI,
    BextI,
}

impl BitManipOp {
    fn name(self) -> &'static str {
        match self {
            Self::Sh1Add => "sh1add",
            Self::Sh2Add => "sh2add",
            Self::Sh3Add => "sh3add",
            Self::AddUw => "add.uw",
            Self::Sh1AddUw => "sh1add.uw",
            Self::Sh2AddUw => "sh2add.uw",
            Self::Sh3AddUw => "sh3add.uw",
            Self::SlliUw => "slli.uw",
            Self::AndN => "andn",
            Self::OrN => "orn",
            Self::Xnor => "xnor",
            Self::Rol => "rol",
            Self::Ror => "ror",
            Self::Rori => "rori",
            Self::RolW => "rolw",
            Self::RorW => "rorw",
            Self::RorIw => "roriw",
            Self::Clz => "clz",
            Self::Ctz => "ctz",
            Self::ClzW => "clzw",
            Self::CtzW => "ctzw",
            Self::Cpop => "cpop",
            Self::CpopW => "cpopw",
            Self::Min => "min",
            Self::MinU => "minu",
            Self::Max => "max",
            Self::MaxU => "maxu",
            Self::SextB => "sext.b",
            Self::SextH => "sext.h",
            Self::ZextH => "zext.h",
            Self::OrcB => "orc.b",
            Self::Rev8 => "rev8",
            Self::Bclr => "bclr",
            Self::Bset => "bset",
            Self::Binv => "binv",
            Self::Bext => "bext",
            Self::BclrI => "bclri",
            Self::BsetI => "bseti",
            Self::BinvI => "binvi",
            Self::BextI => "bexti",
        }
    }
}

/// An RV64B instruction implemented by this extension.
#[derive(Debug, Clone)]
pub(crate) enum Rv64BInstr {
    /// Register-register bit-manipulation operation.
    Reg {
        op: BitManipOp,
        rd: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// Register-immediate or unary bit-manipulation operation.
    Imm {
        op: BitManipOp,
        rd: Reg,
        lhs: Reg,
        imm: u32,
    },
}

impl ExtInstr for Rv64BInstr {
    fn opname(&self) -> &str {
        match self {
            Self::Reg { op, .. } | Self::Imm { op, .. } => op.name(),
        }
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        match self {
            Self::Reg { op, rd, lhs, rhs } => {
                let lhs = parens(ctx.read_var(*lhs));
                let rhs = parens(ctx.read_var(*rhs));
                ctx.write_var(*rd, &reg_expr(*op, &lhs, &rhs));
            }
            Self::Imm { op, rd, lhs, imm } => {
                let lhs = parens(ctx.read_var(*lhs));
                ctx.write_var(*rd, &imm_expr(*op, &lhs, *imm));
            }
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        let dst = match self {
            Self::Reg { rd, .. } | Self::Imm { rd, .. } => *rd,
        };
        CfgEffect::WriteUnknown { dst }
    }
}

fn parens(value: String) -> String {
    format!("({value})")
}

fn reg_expr(op: BitManipOp, lhs: &str, rhs: &str) -> String {
    match op {
        BitManipOp::Sh1Add => format!("({lhs} << 1) + {rhs}"),
        BitManipOp::Sh2Add => format!("({lhs} << 2) + {rhs}"),
        BitManipOp::Sh3Add => format!("({lhs} << 3) + {rhs}"),
        BitManipOp::AddUw => format!("(uint64_t)(uint32_t){lhs} + {rhs}"),
        BitManipOp::Sh1AddUw => format!("(((uint64_t)(uint32_t){lhs}) << 1) + {rhs}"),
        BitManipOp::Sh2AddUw => format!("(((uint64_t)(uint32_t){lhs}) << 2) + {rhs}"),
        BitManipOp::Sh3AddUw => format!("(((uint64_t)(uint32_t){lhs}) << 3) + {rhs}"),
        BitManipOp::AndN => format!("{lhs} & ~{rhs}"),
        BitManipOp::OrN => format!("{lhs} | ~{rhs}"),
        BitManipOp::Xnor => format!("~({lhs} ^ {rhs})"),
        BitManipOp::Rol => format!("rv64b_rol64({lhs}, {rhs})"),
        BitManipOp::Ror => format!("rv64b_ror64({lhs}, {rhs})"),
        BitManipOp::RolW => format!("rv64b_sext32(rv64b_rol32((uint32_t){lhs}, {rhs}))"),
        BitManipOp::RorW => format!("rv64b_sext32(rv64b_ror32((uint32_t){lhs}, {rhs}))"),
        BitManipOp::Min => format!("((int64_t){lhs} < (int64_t){rhs} ? {lhs} : {rhs})"),
        BitManipOp::MinU => format!("({lhs} < {rhs} ? {lhs} : {rhs})"),
        BitManipOp::Max => format!("((int64_t){lhs} > (int64_t){rhs} ? {lhs} : {rhs})"),
        BitManipOp::MaxU => format!("({lhs} > {rhs} ? {lhs} : {rhs})"),
        BitManipOp::Bclr => format!("{lhs} & ~(1ull << ({rhs} & 0x3full))"),
        BitManipOp::Bset => format!("{lhs} | (1ull << ({rhs} & 0x3full))"),
        BitManipOp::Binv => format!("{lhs} ^ (1ull << ({rhs} & 0x3full))"),
        BitManipOp::Bext => format!("({lhs} >> ({rhs} & 0x3full)) & 1ull"),
        _ => unreachable!("invalid RV64B register operation"),
    }
}

fn imm_expr(op: BitManipOp, lhs: &str, imm: u32) -> String {
    let imm = format!("{imm}u");
    match op {
        BitManipOp::SlliUw => format!("((uint64_t)(uint32_t){lhs}) << {imm}"),
        BitManipOp::Rori => format!("rv64b_ror64({lhs}, {imm})"),
        BitManipOp::RorIw => format!("rv64b_sext32(rv64b_ror32((uint32_t){lhs}, {imm}))"),
        BitManipOp::Clz => format!("rv64b_clz64({lhs})"),
        BitManipOp::Ctz => format!("rv64b_ctz64({lhs})"),
        BitManipOp::ClzW => format!("rv64b_clz32((uint32_t){lhs})"),
        BitManipOp::CtzW => format!("rv64b_ctz32((uint32_t){lhs})"),
        BitManipOp::Cpop => format!("rv64b_cpop64({lhs})"),
        BitManipOp::CpopW => format!("rv64b_cpop32((uint32_t){lhs})"),
        BitManipOp::SextB => format!("rv64b_sext8((uint8_t){lhs})"),
        BitManipOp::SextH => format!("rv64b_sext16((uint16_t){lhs})"),
        BitManipOp::ZextH => format!("(uint64_t)(uint16_t){lhs}"),
        BitManipOp::OrcB => format!("rv64b_orc_b({lhs})"),
        BitManipOp::Rev8 => format!("rv64b_rev8({lhs})"),
        BitManipOp::BclrI => format!("{lhs} & ~(1ull << {imm})"),
        BitManipOp::BsetI => format!("{lhs} | (1ull << {imm})"),
        BitManipOp::BinvI => format!("{lhs} ^ (1ull << {imm})"),
        BitManipOp::BextI => format!("({lhs} >> {imm}) & 1ull"),
        _ => unreachable!("invalid RV64B immediate operation"),
    }
}
