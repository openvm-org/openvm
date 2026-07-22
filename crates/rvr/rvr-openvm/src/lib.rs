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
pub use emit::{CProject, EmitContext, InvalidRvrExecutionKind, RvrExecutionKind};
pub use toolchain::{
    default_addr2line_cmd, default_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, default_linker_or_lld, default_make_command, runtime_toolchain, Compiler,
    MissingTool, MissingTools, RuntimeToolchain, RuntimeToolchainError,
};
