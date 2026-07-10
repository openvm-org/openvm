pub mod codegen;
mod context;
mod project;

pub use codegen::instr_emits_inline_record;
pub use context::EmitContext;
pub use project::{CProject, InvalidRvrExecutionKind, RvrExecutionKind};
