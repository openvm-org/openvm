pub mod codegen;
mod context;
mod project;

pub use codegen::{
    inline_record_shape_for_instr, inline_record_shape_for_terminator, instr_emits_inline_record,
    AddIArenaFieldOffsets, Alu3ArenaFieldOffsets, Alu3WArenaFieldOffsets, ArenaNativeGeometry,
    ArenaNativeLayout, Branch2ArenaFieldOffsets, InlineRecordShape, InstrCodegen,
    LoadStoreArenaFieldOffsets, Rw1ArenaFieldOffsets, Wr1ArenaFieldOffsets,
};
pub use context::EmitContext;
pub use project::{
    CProject, G2DsoManifestConfigV2, G2EmissionMode, InvalidRvrExecutionKind, RvrExecutionKind,
};
