//! rvr-openvm backend pieces shared by OpenVM integration.

mod constants;
pub mod emit;
pub mod toolchain;

pub use constants::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
pub use emit::{CProject, EmitContext, InstrCodegen, TracerMode};
use sha2::{Digest, Sha256};
pub use toolchain::{
    default_addr2line_cmd, default_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, default_linker_or_lld, linker_exists, Compiler,
};

/// Cache stamp for backend artifacts used by rvr-native compilation.
///
/// This covers codegen/runtime support living in the `rvr-openvm` backend crate.
#[must_use]
pub fn backend_cache_stamp() -> String {
    let mut hasher = Sha256::new();
    for source in [
        include_str!("emit/context.rs"),
        include_str!("emit/codegen.rs"),
        include_str!("emit/project.rs"),
        include_str!("toolchain.rs"),
        include_str!("../c/openvm_io.c"),
        include_str!("../c/openvm_io.h"),
        include_str!("../c/openvm_state.h"),
        include_str!("../c/openvm_tracer_pure.h"),
        include_str!("../c/openvm_tracer_metered.h"),
        include_str!("../c/openvm_tracer_metered_cost.h"),
        include_str!("../c/rv_muldiv.h"),
        include_str!("../c/rvr_ext_wrappers.c"),
        include_str!("../c/Makefile"),
    ] {
        hasher.update(source.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}
