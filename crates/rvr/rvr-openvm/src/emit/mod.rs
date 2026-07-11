pub mod codegen;
mod context;
mod project;

pub use codegen::{
    inline_record_shape_for_instr, inline_record_shape_for_terminator, instr_emits_inline_record,
    Alu3ArenaFieldOffsets, ArenaNativeGeometry, ArenaNativeLayout, Branch2ArenaFieldOffsets,
};
pub use context::EmitContext;
pub use project::{CProject, InvalidRvrExecutionKind, RvrExecutionKind};
pub use rvr_openvm_ir::InlineRecordShape;
