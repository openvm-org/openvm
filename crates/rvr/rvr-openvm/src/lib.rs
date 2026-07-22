//! Native RVR backend for OpenVM.
//!
//! The current implementation supports RV64 artifacts.
//!
//! # Runtime Requirements
//!
//! RVR emits generated C and builds a native dynamic library before execution.
//! The environment must provide the toolchain selected by [`runtime_toolchain`]:
//! a C compiler, an LLVM linker, and `make`.

mod constants;
pub mod emit;
pub mod toolchain;

pub use constants::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
pub use emit::{
    inline_record_shape_for_instr, inline_record_shape_for_terminator,
    instr_emits_inline_record, AddIArenaFieldOffsets, Alu3ArenaFieldOffsets,
    Alu3WArenaFieldOffsets, AluImmArenaFieldOffsets, ArenaNativeGeometry, ArenaNativeLayout,
    Branch2ArenaFieldOffsets, CProject, EmitContext, G2DsoManifestConfigV2, G2EmissionMode,
    InlineRecordShape, InvalidRvrExecutionKind, LoadStoreArenaFieldOffsets, RvrExecutionKind,
    Rw1ArenaFieldOffsets, Wr1ArenaFieldOffsets,
};
pub use toolchain::{
    default_addr2line_cmd, default_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, default_linker_or_lld, default_make_command, runtime_toolchain, Compiler,
    MissingTool, MissingTools, RuntimeToolchain, RuntimeToolchainError,
};
