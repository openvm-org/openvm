//! rvr-openvm backend pieces shared by OpenVM integration.

mod constants;
pub mod emit;
pub mod toolchain;

pub use constants::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
pub use emit::{CProject, EmitContext, InstrCodegen, TracerMode};
pub use toolchain::{
    default_addr2line_cmd, default_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, default_linker_or_lld, linker_exists, Compiler,
};
