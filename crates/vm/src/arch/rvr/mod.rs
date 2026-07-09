//! OpenVM-owned rvr integration layer.

mod abi_consts;
pub mod bridge;
pub mod compile;
pub mod debug;
pub mod execute;
pub mod io;
pub mod log_native;
pub mod metered;
pub mod metered_cost;
pub mod preflight;
pub mod preflight_normalizer;
pub mod pure;
pub mod state;

pub use compile::{
    build_pc_to_chip, classify_preflight_opcodes, classify_preflight_opcodes_with_extensions,
    compile, compile_metered, compile_metered_cost, compile_metered_segment_boundary,
    compile_preflight, compile_preflight_with_extensions, compile_with_options,
    load_compiled_from_path, ChipMapping, CompileError, CompileOptions, RvrCompiled,
    RvrPreflightOpcodeClass,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::{
    execute, execute_metered, execute_metered_cost, execute_metered_segment_boundary,
    execute_preflight, rv_execute, ExecuteError, RvrPreflightRunResult,
};
pub use log_native::{
    generate_record_arenas_from_logs, LogNativeAccessView, LogNativeAssembler,
    LogNativeAssemblerRegistry, LogNativeOpcodeAdmitter, VmRvrLogNativeExtension,
};
pub use metered::{
    RunToCompletion, RvrMeteredInstance, RvrMeteredInstanceWith, RvrMeteredResult,
    RvrMeteredSegmentInstance, SegmentBoundary,
};
pub use metered_cost::{
    MeteredCostData, MeteredCostMeter, PureTracer, PureTracerData, RvrMeteredCostInstance,
    RvrMeteredCostResult,
};
pub use preflight::{
    MemoryLogEntry, PreflightRawLogs, PreflightTracer, PreflightTracerData, ProgramLogEntry,
    RvrPreflightInstance, RvrPreflightOutput, RvrPreflightRoute, PREFLIGHT_INITIAL_TIMESTAMP,
    PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE,
    PREFLIGHT_TRACER_KIND,
};
pub use preflight_normalizer::{
    normalize_preflight_memory_logs, PreflightMemoryAccessAux, PreflightMemoryReplay,
    PreflightNormalizeError,
};
pub use pure::{RvrPureInstance, RvrPureResult};
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, runtime_toolchain, RuntimeToolchain, RuntimeToolchainError, SuspendPolicy,
    TracerMode,
};

pub use crate::arch::execution_mode::metered::segment_ctx::SegmentationLimits;
