//! OpenVM-owned rvr integration layer.

mod abi_consts;
pub mod bridge;
pub mod compile;
pub mod debug;
pub mod execute;
pub mod io;
pub mod metered;
pub mod metered_cost;
pub mod pure;
pub mod state;

pub use compile::{
    build_pc_to_chip, compile, compile_metered, compile_metered_cost,
    compile_metered_segment_boundary, load_compiled_from_path, ChipMapping, CompileError,
    RvrCompiled,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::{
    build_callbacks, execute, execute_metered, execute_metered_cost,
    execute_metered_segment_boundary, register_openvm_callbacks, rv_execute, ExecuteError,
};
pub use metered::{
    RunToCompletion, RvrMeteredInstance, RvrMeteredInstanceWith, RvrMeteredResult,
    RvrMeteredSegmentInstance, SegmentBoundary,
};
pub use metered_cost::{
    MeteredCostData, MeteredCostMeter, PureTracer, PureTracerData, RvrMeteredCostInstance,
    RvrMeteredCostResult,
};
pub use pure::{RvrPureInstance, RvrPureResult};
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, runtime_toolchain, RuntimeToolchain, RuntimeToolchainError, SuspendPolicy,
    TracerMode,
};

pub use crate::arch::execution_mode::metered::segment_ctx::SegmentationLimits;
