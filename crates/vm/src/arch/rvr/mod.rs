//! OpenVM-owned rvr integration layer.

mod abi_consts;
pub mod bridge;
pub mod compile;
pub mod debug;
mod execute;
mod initial_image;
pub mod io;
pub mod log_native;
pub mod metered;
pub mod metered_cost;
pub mod preflight;
pub mod preflight_normalizer;
pub mod preflight_pool;
pub mod pure;
pub mod state;

pub use compile::{
    build_pc_to_chip, compile, compile_metered, compile_metered_cost,
    compile_metered_segment_boundary, compile_preflight, compile_preflight_with_extensions,
    compile_with_instret_tracking, compile_with_options,
    load_compiled_from_path, ChipMapping, CompileError, CompileOptions, RvrCompiled,
    RvrDeltaDecodePrecompute, RvrInlineRecordsMeta, RvrPreflightOpcodeClass,
};
#[cfg(any(test, feature = "test-utils"))]
pub use compile::{
    preflight_compile_invocations_for_test, reset_preflight_compile_invocations_for_test,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::{
    execute, execute_metered, execute_metered_cost, execute_metered_segment_boundary,
    execute_preflight, rv_execute, ExecuteError, RvrPreflightRunResult,
};
pub use initial_image::RvrInitialImage;
pub use log_native::{
    generate_record_arenas_from_logs, generate_record_arenas_from_logs_with_compact,
    Alu3ArenaFieldOffsets, Alu3WArenaFieldOffsets, ArenaNativeGeometry, ArenaNativeLayout,
    Branch2ArenaFieldOffsets, DeltaAccessPattern, LoadStoreArenaFieldOffsets, LogNativeAccessView,
    LogNativeAssembler, LogNativeAssemblerRegistry, LogNativeInlineAssembler,
    LogNativeOpcodeAdmitter, RvrDeltaDecodeEntry, RvrDeltaDecodeInfo, Rw1ArenaFieldOffsets,
    VmRvrLogNativeExtension, Wr1ArenaFieldOffsets,
};
pub use metered::{RvrMeteredExecutionOutcome, RvrMeteredInstance, RvrMeteredSegmentInstance};
pub use metered_cost::{MeteredCostState, RvrMeteredCostInstance};
pub use pure::{
    RvrPureInstance, RvrPureWithInstretTrackingInstance, RvrTrackedExecution,
    RvrTrackedExecutionOutcome,
};
pub use preflight::{
    rvr_preflight_engine_env_override, ChipRecordBuf, MemoryLogEntry, PreflightRawLogs,
    PreflightTracer, PreflightTracerData, ProgramLogEntry, RvrDeltaRecords, RvrInlineChipRecords,
    RvrPreflightEngine, RvrPreflightInstance, RvrPreflightOutput, RvrPreflightRoute, TouchedBlock,
    PREFLIGHT_ADDSUB_RECORD_SIZE, PREFLIGHT_BRANCH2_RECORD_SIZE,
    PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL, PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW,
    PREFLIGHT_CHIP_RECORD_FLAG_RESIDUAL_MEMORY_CHRONOLOGY,
    PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS, PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE,
    PREFLIGHT_DELTA_RECORD_SIZE, PREFLIGHT_INITIAL_TIMESTAMP, PREFLIGHT_MEMORY_KIND_READ,
    PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE, PREFLIGHT_RW1_RECORD_SIZE,
    PREFLIGHT_TRACER_KIND, PREFLIGHT_WR1_RECORD_SIZE,
};
pub use preflight_normalizer::{
    build_preflight_replay, PreflightMemoryAccessAux, PreflightMemoryReplay,
    PreflightNormalizeError, PreflightShadowsView,
};
pub use preflight_pool::RvrPreflightBufferPool;
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, runtime_toolchain, RuntimeToolchain, RuntimeToolchainError, RvrExecutionKind,
};

pub use crate::arch::execution_mode::metered::segment_ctx::SegmentationLimits;
