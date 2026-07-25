//! OpenVM-owned RVR execution.
//!
//! The execution modes share opcode semantics but produce different outputs:
//!
//! ```text
//! pure        program + mutable VM state -> final VM state
//! metered     program + mutable VM state -> final VM state + segment plan
//! checkpoint  program + mutable VM state -> final VM state + checkpoints + residuals
//! ```
//!
//! Checkpoint preflight is the proving path. Its two append-only arrays are a
//! compact replay seed, not chip records. GPU expansion re-executes independent
//! checkpoint intervals to derive the immutable program and memory history used
//! by system, RISC-V, and extension trace generators.

mod abi_consts;
pub mod bridge;
mod cache;
pub mod checkpoint_preflight;
pub mod compile;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod debug;
mod execute;
mod initial_image;
pub mod io;
pub mod metered;
pub mod metered_cost;
#[cfg(all(feature = "cuda", any(test, feature = "test-utils")))]
mod postflight;
pub mod preflight;
pub mod pure;
pub mod state;

pub use checkpoint_preflight::{
    RvrCheckpointPreflightExecution, RvrCheckpointPreflightInstance, RvrCheckpointPreflightLimits,
    RvrCheckpointPreflightTranscript,
};
pub use compile::{
    build_pc_to_chip, compile, compile_checkpoint_preflight, compile_metered, compile_metered_cost,
    compile_metered_segment_boundary, compile_preflight, compile_with_instret_tracking,
    compile_with_options, load_compiled_from_path, ChipMapping, CompileError, CompileOptions,
    RvrCompiled,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::ExecuteError;
pub use initial_image::RvrInitialImage;
pub use metered::{RvrMeteredExecutionOutcome, RvrMeteredInstance, RvrMeteredSegmentInstance};
pub use metered_cost::{MeteredCostState, RvrMeteredCostInstance};
pub use preflight::{
    RvrPreflightEndpoint, RvrPreflightExecution, RvrPreflightInstance, RvrPreflightLimits,
    RvrPreflightTranscript,
};
pub use pure::{
    RvrPureInstance, RvrPureWithInstretTrackingInstance, RvrTrackedExecution,
    RvrTrackedExecutionOutcome,
};
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, runtime_toolchain, RuntimeToolchain, RuntimeToolchainError, RvrExecutionKind,
};

pub use crate::arch::execution_mode::metered::segment_ctx::SegmentationLimits;
