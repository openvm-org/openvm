//! OpenVM-owned rvr integration layer.

pub mod compile;
pub mod debug;
pub mod execute;
pub mod io;
pub mod metered;
pub mod metered_cost;
pub mod state;

pub use compile::{
    compile, compile_metered, compile_metered_cost, compile_metered_cost_with_extensions,
    compile_metered_cost_with_limit, compile_metered_with_extensions, compile_with_extensions,
    compile_with_limit, compile_with_options, load_compiled_from_path, ChipMapping, CompileError,
    CompileOptions, RvrCompiled,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::{
    build_callbacks, build_io_state, execute, execute_metered, execute_metered_cost,
    execute_metered_cost_with_limit, execute_with_limit, register_and_execute, ExecuteError,
    RvrExecutionResult, RvrLimitedResult, RvrMeteredCostLimitedResult, RvrMeteredCostResult,
};
pub use io::DeferralData;
pub use metered::{build_metered_config, MeteredConfig, RvrMeteredResult, RvrSegment};
pub use metered_cost::{
    build_metered_cost_config, MeteredCostConfig, MeteredCostData, MeteredCostMeter, PureTracer,
    PureTracerData,
};
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, TracerMode,
};

pub use crate::arch::execution_mode::metered::segment_ctx::{
    SegmentationConfig, SegmentationLimits,
};
