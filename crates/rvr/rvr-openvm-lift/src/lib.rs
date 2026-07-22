//! rvr-openvm-lift: Convert OpenVM VmExe to rvr-openvm-ir types.
//!
//! This crate provides the bridge between OpenVM's instruction format
//! and the shared rvr-openvm-ir intermediate representation.
//!
//! The `opcode` module lifts individual OpenVM instructions to `LiftedInstr`,
//! which is either a body `ExtInstr` or a `Terminator` (control flow).
//! The `convert` module provides the top-level `convert_vmexe_to_ir` function.
//! The `cfg` module constructs control-flow graphs and propagates constants.
//! The `extension` module provides the `RvrExtension` trait and `ExtensionRegistry`.

pub mod cfg;
pub mod convert;
pub mod extension;
pub mod instruction;
pub mod opcode;

pub use cfg::{build_blocks, CfgError};
pub use convert::{convert_vmexe_to_ir, convert_vmexe_to_ir_with_debug, ConvertError};
pub use extension::{
    air_index_to_c, decode_variable, fixed_trace_rows_for_chip,
    max_main_memory_pages_for_contiguous_range, opcode_air_idx, AirIndex, ExtensionError,
    ExtensionRegistry, RvrExtension, RvrExtensionCtx, RvrExtensions, RvrRuntimeExtension,
    TraceChipIndex, VmRvrExtension, MAIN_MEMORY_PAGE_BYTES,
};
pub use instruction::RvrInstruction;
