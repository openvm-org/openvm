use serde::{Deserialize, Serialize};

use crate::instr::{Instr, Reg};

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

/// Branch condition (used in Terminator::Branch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

/// Block terminator — control flow at the end of a basic block.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Fall through to pc + 4 (implicit next block).
    FallThrough,
    /// Static jump (JAL). `link_rd` writes pc+4 to rd before jumping.
    Jump { link_rd: Option<Reg>, target: u32 },
    /// Dynamic jump (JALR). `resolved` filled in by CFG analysis.
    JumpDyn {
        link_rd: Option<Reg>,
        rs1: Reg,
        imm: i32,
        resolved: Vec<u32>,
    },
    /// Conditional branch. Falls through if false.
    Branch {
        cond: BranchCond,
        rs1: Reg,
        rs2: Reg,
        target: u32,
    },
    /// Program exit.
    Exit { code: u32 },
    /// Illegal instruction / debug panic.
    Trap { message: String },
    /// Extension terminator (from a registered extension crate).
    /// Carries its own codegen via `ExtInstr::emit_c`.
    Extension(Box<dyn crate::ExtInstr>),
}

impl Terminator {
    /// Returns the set of possible successor PCs (for CFG building).
    pub fn successors(&self, fall_pc: u32) -> Vec<u32> {
        match self {
            Terminator::FallThrough => vec![fall_pc],
            Terminator::Jump { target, .. } => vec![*target],
            Terminator::JumpDyn { resolved, .. } => resolved.clone(),
            Terminator::Branch { target, .. } => vec![*target, fall_pc],
            Terminator::Exit { .. } | Terminator::Trap { .. } => vec![],
            Terminator::Extension(ext) => ext.successors(fall_pc),
        }
    }

    pub fn opname(&self) -> &str {
        match self {
            Terminator::FallThrough => "fallthrough",
            Terminator::Jump {
                link_rd: Some(_), ..
            } => "jal",
            Terminator::Jump { link_rd: None, .. } => "j",
            Terminator::JumpDyn {
                link_rd: Some(_), ..
            } => "jalr",
            Terminator::JumpDyn { link_rd: None, .. } => "jr",
            Terminator::Branch { cond, .. } => match cond {
                BranchCond::Eq => "beq",
                BranchCond::Ne => "bne",
                BranchCond::Lt => "blt",
                BranchCond::Ge => "bge",
                BranchCond::Ltu => "bltu",
                BranchCond::Geu => "bgeu",
            },
            Terminator::Exit { .. } => "exit",
            Terminator::Trap { .. } => "trap",
            Terminator::Extension(ext) => ext.opname(),
        }
    }

    /// Returns true if this is a block-ending terminator (not fall-through).
    pub fn is_block_end(&self) -> bool {
        match self {
            Terminator::FallThrough => false,
            Terminator::Extension(ext) => ext.is_block_end(),
            _ => true,
        }
    }
}

/// An instruction at a specific PC.
#[derive(Debug, Clone)]
pub struct InstrAt {
    pub pc: u32,
    pub instr: Instr,
    /// Source location from guest ELF debug info.
    pub source_loc: Option<SourceLoc>,
}

/// A lifted instruction that may be a body instruction or a terminator.
/// Used as intermediate output from the lifter before block construction.
#[derive(Debug, Clone)]
pub enum LiftedInstr {
    Body(InstrAt),
    Term {
        pc: u32,
        terminator: Terminator,
        source_loc: Option<SourceLoc>,
    },
}

impl LiftedInstr {
    pub fn pc(&self) -> u32 {
        match self {
            LiftedInstr::Body(i) => i.pc,
            LiftedInstr::Term { pc, .. } => *pc,
        }
    }
}

/// A basic block.
#[derive(Debug, Clone)]
pub struct Block {
    pub start_pc: u32,
    /// End PC (exclusive).
    pub end_pc: u32,
    /// Body instructions (no branches/jumps).
    pub instructions: Vec<InstrAt>,
    /// Control flow at the end.
    pub terminator: Terminator,
    /// PC of the terminating instruction.
    pub terminator_pc: u32,
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
