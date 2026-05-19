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
    compile, compile_metered, compile_metered_cost, load_compiled_from_path, ChipMapping,
    CompileError, RvrCompiled,
};
pub use debug::{default_addr2line_cmd, GuestDebugMap};
pub use execute::{
    build_callbacks, execute, execute_metered, execute_metered_cost, register_openvm_callbacks,
    rv_execute, ExecuteError,
};
pub use metered::{build_pc_to_chip, RvrMeteredInstance};
pub use metered_cost::{
    build_metered_cost_config, MeteredCostConfig, MeteredCostData, MeteredCostMeter, PureTracer,
    PureTracerData, RvrMeteredCostInstance, RvrMeteredCostResult,
};
pub use pure::{RvrPureInstance, RvrPureResult};
pub use rvr_openvm::{
    default_compiler as default_native_compiler, default_compiler_command, default_dwarfdump_cmd,
    default_linker, TracerMode,
};

pub use crate::arch::execution_mode::metered::segment_ctx::{
    SegmentationConfig, SegmentationLimits,
};
