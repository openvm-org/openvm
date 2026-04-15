//! rvr-openvm-lift: Convert OpenVM VmExe to rvr-openvm-ir types.
//!
//! This crate provides the bridge between OpenVM's instruction format
//! and the rvr-openvm-ir intermediate representation for RV32IM instructions.
//!
//! The `opcode` module lifts individual OpenVM instructions to `LiftedInstr`,
//! which is either a body `Instr` or a `Terminator` (control flow).
//! The `convert` module provides the top-level `convert_vmexe_to_ir` function.
//! The `cfg` module builds basic blocks from flat IR using rvr-openvm-ir types.
//! The `extension` module provides the `RvrExtension` trait and `ExtensionRegistry`.

pub mod cfg;
pub mod convert;
pub mod extension;
pub mod helpers;
pub mod opcode;

pub use cfg::build_blocks;
pub use convert::{
    convert_vmexe_to_ir, convert_vmexe_to_ir_with_debug, scan_init_memory_for_code_pointers,
    ConvertError,
};
pub use extension::{ExtensionRegistry, RvrExtension};
pub use helpers::{decode_reg, resolve_opcode_air_idx};
