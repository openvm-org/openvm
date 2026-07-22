use serde::{Deserialize, Serialize};

use crate::{CfgTerm, ExtInstr};

/// Source location for debug info (`#line` directives).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SourceLoc {
    /// Source file path.
    pub file: String,
    /// Line number.
    pub line: u32,
    /// Function name (for comments; may be empty).
    pub function: String,
}

impl SourceLoc {
    pub fn new(file: &str, line: u32, function: &str) -> Self {
        Self {
            file: file.to_string(),
            line,
            function: function.to_string(),
        }
    }

    /// A source location is valid if it has a non-empty, non-placeholder file and a positive line.
    pub fn is_valid(&self) -> bool {
        !self.file.is_empty() && self.file != "??" && self.line > 0
    }
}

/// Block terminator — control flow at the end of a basic block.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Fall through to pc + 4 (implicit next block).
    FallThrough,
    /// Program exit.
    Exit { code: u32 },
    /// Illegal instruction / debug panic.
    Trap { message: String },
    /// Control-flow instruction owned by a registered extension.
    Extension(Box<dyn ExtInstr>),
}

impl Terminator {
    /// Returns the target-neutral control-flow behavior of this terminator.
    pub fn cfg_term(&self, pc: u64, fall_pc: u64) -> CfgTerm {
        match self {
            Self::FallThrough => CfgTerm::FallThrough,
            Self::Exit { code } => CfgTerm::Exit { code: *code },
            Self::Trap { message } => CfgTerm::Trap {
                message: message.clone(),
            },
            Self::Extension(instr) => instr.cfg_term(pc, fall_pc).unwrap_or(CfgTerm::FallThrough),
        }
    }

    /// Returns the set of possible successor PCs for CFG construction.
    pub fn successors(&self, pc: u64, fall_pc: u64) -> Vec<u64> {
        match self.cfg_term(pc, fall_pc) {
            CfgTerm::FallThrough => vec![fall_pc],
            CfgTerm::Jump { target, .. } => vec![target],
            CfgTerm::JumpIndirect { resolved, .. } => resolved,
            CfgTerm::Branch { target, .. } => vec![target, fall_pc],
            CfgTerm::Exit { .. } | CfgTerm::Trap { .. } => Vec::new(),
            CfgTerm::Opaque { successors } => successors,
        }
    }

    pub fn opname(&self) -> &str {
        match self {
            Self::FallThrough => "fallthrough",
            Self::Exit { .. } => "exit",
            Self::Trap { .. } => "trap",
            Self::Extension(instr) => instr.opname(),
        }
    }

    /// Returns true if this is a block-ending terminator rather than fall-through.
    pub fn is_block_end(&self, pc: u64, fall_pc: u64) -> bool {
        !matches!(self.cfg_term(pc, fall_pc), CfgTerm::FallThrough)
    }
}

/// An instruction at a specific PC.
#[derive(Debug, Clone)]
pub struct InstrAt {
    pub pc: u64,
    pub instr: Box<dyn ExtInstr>,
    /// Source location from guest ELF debug info.
    pub source_loc: Option<SourceLoc>,
}

/// A lifted instruction that may be a body instruction or a terminator.
/// Used as intermediate output from the lifter before block construction.
#[derive(Debug, Clone)]
pub enum LiftedInstr {
    Body(InstrAt),
    Term {
        pc: u64,
        terminator: Terminator,
        source_loc: Option<SourceLoc>,
    },
}

impl LiftedInstr {
    pub fn pc(&self) -> u64 {
        match self {
            Self::Body(i) => i.pc,
            Self::Term { pc, .. } => *pc,
        }
    }
}

/// A basic block.
#[derive(Debug, Clone)]
pub struct Block {
    pub start_pc: u64,
    /// End PC (exclusive).
    pub end_pc: u64,
    /// Body instructions (no branches/jumps).
    pub instructions: Vec<InstrAt>,
    /// Control flow at the end.
    pub terminator: Terminator,
    /// PC of the terminating instruction.
    pub terminator_pc: u64,
    /// Source location of the terminating instruction.
    pub terminator_source_loc: Option<SourceLoc>,
}

impl Block {
    pub fn insn_count(&self) -> u32 {
        // Body instructions + 1 for the terminator instruction.
        // FallThrough has no real terminator instruction — it's a synthetic
        // block boundary, so don't add 1.
        let base = self.instructions.len() as u32;
        if matches!(self.terminator, Terminator::FallThrough) {
            base
        } else {
            base + 1
        }
    }
}
