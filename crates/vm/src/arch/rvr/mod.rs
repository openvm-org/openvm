//! OpenVM-owned RVR execution.
//!
//! The execution modes share opcode semantics but produce different outputs:
//!
//! ```text
//! pure        program + mutable VM state -> final VM state
//! metered     program + mutable VM state -> final VM state + segment plan
//! preflight   program + mutable VM state -> final VM state + replay seed
//! ```
//!
//! Preflight emits two append-only arrays containing checkpoints and replay values.
//! They are a compact replay seed, not chip records. Postflight re-executes
//! independent intervals on the GPU to derive the immutable program and memory
//! history used by system, RISC-V, and extension trace generators.

mod abi_consts;
pub mod bridge;
mod cache;
pub mod compile;
pub mod debug;
mod execute;
mod initial_image;
pub mod io;
pub mod metered;
pub mod metered_cost;
mod preflight;
pub mod pure;
pub mod state;

pub use compile::{
    build_pc_to_chip, compile, compile_metered, compile_metered_cost,
    compile_metered_segment_boundary, compile_with_instret_tracking, compile_with_options,
    load_compiled_from_path, ChipMapping, CompileError, CompileOptions, RvrCompiled,
    RvrProgramMetadata,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::ExecuteError;
pub use initial_image::RvrInitialImage;
pub use metered::{RvrMeteredExecutionOutcome, RvrMeteredInstance, RvrMeteredSegmentInstance};
pub use metered_cost::{MeteredCostState, RvrMeteredCostInstance};
pub use openvm_instructions::exe::CfgHints;
pub use preflight::{
    PreflightEndpoint, PreflightExecution, PreflightInstance, PreflightLimits, PreflightTranscript,
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
