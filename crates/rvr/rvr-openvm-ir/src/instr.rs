use crate::{EmitCtx, FixedTraceRows};

/// Opaque value location used by CFG analysis and generated-code access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueSlot(u32);

impl ValueSlot {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Memory access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemWidth {
    Byte,
    Half,
    Word,
    Double,
}

impl MemWidth {
    pub const fn bytes(self) -> u8 {
        match self {
            Self::Byte => 1,
            Self::Half => 2,
            Self::Word => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgOperand {
    Slot(ValueSlot),
    Const(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    LessThanSigned,
    LessThanUnsigned,
    Mul,
    MulHighSigned,
    MulHighSignedUnsigned,
    MulHighUnsigned,
    DivSigned,
    DivUnsigned,
    RemSigned,
    RemUnsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgResultWidth {
    U32,
    U64,
    SignExtend32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgIntWidth {
    U32,
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgEffect {
    None,
    WriteUnknown {
        dst: ValueSlot,
    },
    WriteConst {
        dst: ValueSlot,
        value: u64,
    },
    WriteOp {
        dst: ValueSlot,
        op: CfgOp,
        lhs: CfgOperand,
        rhs: CfgOperand,
        result: CfgResultWidth,
    },
    /// Forget every tracked value when an instruction does not describe its effects.
    ClobberAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgBranchCond {
    Eq,
    Ne,
    LessThanSigned,
    GreaterEqualSigned,
    LessThanUnsigned,
    GreaterEqualUnsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgJumpKind {
    Jump,
    Call,
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgTerm {
    FallThrough,
    Jump {
        kind: CfgJumpKind,
        link_dst: Option<ValueSlot>,
        target: u64,
    },
    JumpIndirect {
        kind: CfgJumpKind,
        link_dst: Option<ValueSlot>,
        base: ValueSlot,
        base_value: CfgOperand,
        offset: i32,
        target_mask: u64,
        resolved: Vec<u64>,
    },
    Branch {
        cond: CfgBranchCond,
        width: CfgIntWidth,
        lhs: ValueSlot,
        rhs: ValueSlot,
        target: u64,
        known: Option<bool>,
    },
    Exit {
        code: u32,
    },
    Trap {
        message: String,
    },
    Opaque {
        successors: Vec<u64>,
    },
}

/// A self-contained instruction node owned by an RVR extension.
pub trait Instr: std::fmt::Debug + Send + Sync {
    fn emit_c(&self, ctx: &mut dyn EmitCtx);

    fn emit_c_term(&self, ctx: &mut dyn EmitCtx, _branch_to: &dyn Fn(u64) -> String) {
        self.emit_c(ctx);
    }

    fn opname(&self) -> &str {
        "instr"
    }

    /// Data-flow behavior used by CFG analysis.
    ///
    /// Implementations must choose an explicit effect so value-slot writes cannot
    /// accidentally preserve stale control-flow constants.
    fn cfg_effect(&self) -> CfgEffect;

    fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
        None
    }

    fn accesses_memory(&self) -> bool {
        true
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        Vec::new()
    }

    fn clone_box(&self) -> Box<dyn Instr>;

    fn with_resolved_jumps(&self, resolved: Vec<u64>) -> Box<dyn Instr> {
        assert!(
            resolved.is_empty(),
            "instruction {} does not accept resolved jump targets",
            self.opname()
        );
        self.clone_box()
    }
}

impl Clone for Box<dyn Instr> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
