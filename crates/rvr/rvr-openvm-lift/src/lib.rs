//! rvr-openvm-lift: Convert OpenVM VmExe to rvr-openvm-ir types.
//!
//! This crate provides the bridge between OpenVM's instruction format
//! and the rvr-openvm-ir intermediate representation for RISC-V instructions.
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
pub mod instruction;
pub mod opcode;

pub use cfg::{build_blocks, CfgError};
pub use convert::{
    convert_vmexe_to_ir, convert_vmexe_to_ir_with_debug, scan_init_memory_for_code_pointers,
    ConvertError,
};
pub use extension::{
    air_index_to_c, fixed_trace_rows_for_chip, opcode_air_idx, AirIndex, ExtensionError,
    ExtensionRegistry, RvrExtension, RvrExtensionCtx, RvrExtensions, RvrRuntimeExtension,
    TraceChipIndex, VmRvrExtension,
};
pub use helpers::{decode_imm_cg, decode_reg};
pub use instruction::RvrInstruction;
