use serde::{Deserialize, Serialize};

use crate::{CfgTerm, Instr};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SourceLoc {
    pub file: String,
    pub line: u32,
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

    pub fn is_valid(&self) -> bool {
        !self.file.is_empty() && self.file != "??" && self.line > 0
    }
}

#[derive(Debug, Clone)]
pub enum Terminator {
    FallThrough,
    Exit { code: u32 },
    Trap { message: String },
    Instr(Box<dyn Instr>),
}

impl Terminator {
    pub fn cfg_term(&self, pc: u64, fall_pc: u64) -> CfgTerm {
        match self {
            Self::FallThrough => CfgTerm::FallThrough,
            Self::Exit { code } => CfgTerm::Exit { code: *code },
            Self::Trap { message } => CfgTerm::Trap {
                message: message.clone(),
            },
            Self::Instr(instr) => instr.cfg_term(pc, fall_pc).unwrap_or(CfgTerm::FallThrough),
        }
    }

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
            Self::Instr(instr) => instr.opname(),
        }
    }

    pub fn is_block_end(&self, pc: u64, fall_pc: u64) -> bool {
        !matches!(self.cfg_term(pc, fall_pc), CfgTerm::FallThrough)
    }
}

#[derive(Debug, Clone)]
pub struct InstrAt {
    pub pc: u64,
    pub instr: Box<dyn Instr>,
    pub source_loc: Option<SourceLoc>,
}

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

#[derive(Debug, Clone)]
pub struct Block {
    pub start_pc: u64,
    pub end_pc: u64,
    pub instructions: Vec<InstrAt>,
    pub terminator: Terminator,
    pub terminator_pc: u64,
    pub terminator_source_loc: Option<SourceLoc>,
}

impl Block {
    pub fn insn_count(&self) -> u32 {
        let base = self.instructions.len() as u32;
        if matches!(self.terminator, Terminator::FallThrough) {
            base
        } else {
            base + 1
        }
    }
}
