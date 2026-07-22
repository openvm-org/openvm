use crate::{ExtEmitCtx, FixedTraceRows};

/// Opaque target-defined mutable state location used by CFG analysis and code generation.
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

/// Operand used by target-neutral CFG evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgOperand {
    /// Value read from an opaque target-defined slot.
    Slot(ValueSlot),
    /// Immediate constant.
    Const(u64),
}

/// Integer operation understood by CFG constant propagation.
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

/// Width and extension behavior of a CFG operation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgResultWidth {
    /// Keep the low 32 bits and zero-extend them.
    U32,
    /// Keep all 64 bits.
    U64,
    /// Keep the low 32 bits and sign-extend them to 64 bits.
    SignExtend32,
}

/// Integer width used to compare branch operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgIntWidth {
    /// Compare the low 32 bits.
    U32,
    /// Compare all 64 bits.
    U64,
}

/// Data-flow effect of one instruction on tracked value slots.
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

/// Target-neutral conditional branch predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgBranchCond {
    Eq,
    Ne,
    LessThanSigned,
    GreaterEqualSigned,
    LessThanUnsigned,
    GreaterEqualUnsigned,
}

/// Control-flow role of a jump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgJumpKind {
    Jump,
    Call,
    Return,
}

/// Target-neutral control-flow behavior used by CFG analysis and code generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgTerm {
    /// Continue at the next instruction.
    FallThrough,
    /// Jump to a statically known target.
    Jump {
        kind: CfgJumpKind,
        link_dst: Option<ValueSlot>,
        target: u64,
    },
    /// Jump to a target computed from an operand and signed offset.
    JumpIndirect {
        kind: CfgJumpKind,
        link_dst: Option<ValueSlot>,
        base_value: CfgOperand,
        offset: i32,
        target_mask: u64,
        resolved: Vec<u64>,
    },
    /// Conditionally jump to `target`, otherwise fall through.
    Branch {
        cond: CfgBranchCond,
        width: CfgIntWidth,
        lhs: ValueSlot,
        rhs: ValueSlot,
        target: u64,
        known: Option<bool>,
    },
    /// Terminate execution with an exit code.
    Exit { code: u32 },
    /// Trap execution with a diagnostic message.
    Trap { message: String },
    /// Instruction-defined control flow with an explicit successor set.
    Opaque { successors: Vec<u64> },
}

/// Trait for self-contained instruction nodes owned by RVR extensions.
pub trait ExtInstr: std::fmt::Debug + Send + Sync {
    /// Emit C code for this instruction through the mode-aware context.
    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx);

    /// Emit C code for an instruction-owned terminator.
    ///
    /// `branch_to(target_pc)` returns the C tail-call statement for a static
    /// successor. The default delegates to `emit_c`.
    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, _branch_to: &dyn Fn(u64) -> String) {
        self.emit_c(ctx);
    }

    /// Short operation name used in generated C comments.
    fn opname(&self) -> &str {
        "ext"
    }

    /// Data-flow behavior used by CFG analysis.
    ///
    /// Implementations must choose an explicit effect so value-slot writes cannot
    /// accidentally preserve stale control-flow constants.
    fn cfg_effect(&self) -> CfgEffect;

    /// Control-flow behavior, if this instruction ends a basic block.
    fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
        None
    }

    /// Whether this instruction may access the target's main guest memory.
    ///
    /// Code generation uses this to decide whether a metered block needs main
    /// memory page tracking. The conservative default is `true`.
    fn accesses_memory(&self) -> bool {
        true
    }

    /// Extra chip rows whose count is known when the artifact is generated.
    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        Vec::new()
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ExtInstr>;

    /// Return a copy with CFG-resolved indirect-jump targets.
    ///
    /// Instructions without indirect control flow reject nonempty targets so
    /// resolved successors cannot be silently discarded.
    fn with_resolved_jumps(&self, resolved: Vec<u64>) -> Box<dyn ExtInstr> {
        assert!(
            resolved.is_empty(),
            "instruction {} does not accept resolved jump targets",
            self.opname()
        );
        self.clone_box()
    }
}

impl Clone for Box<dyn ExtInstr> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
