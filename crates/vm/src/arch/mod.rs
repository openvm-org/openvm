mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
/// Streams-like deferral state
pub mod deferral;
/// Instruction execution traits and types.
/// Execution bus and interface.
pub mod execution;
#[cfg(feature = "metrics")]
pub(crate) mod execution_metrics;
/// Execution context types for different execution modes.
pub mod execution_mode;
mod extensions;
mod hint_stream;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
mod postflight;
mod preflight;
#[cfg(feature = "rvr")]
pub mod rvr;
/// Continuation proving expressed as a scheduler graph.
mod segment_scheduler;
/// VM state definitions
mod state;
/// Top level [VmExecutor] and [VirtualMachine] constructor and API.
pub mod vm;

pub mod hasher;
/// Interpreter for pure and metered VM execution
pub mod interpreter;
/// Interpreter for preflight VM execution, for trace generation purposes.
pub mod interpreter_preflight;
/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use config::*;
pub use execution::*;
pub use execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait};
pub use extensions::*;
pub use hint_stream::HintStream;
pub use integration_api::*;
pub use interpreter::InterpretedInstance;
pub use openvm_circuit_derive::create_handler;
pub use openvm_instructions as instructions;
pub use postflight::{Postflight, PostflightError, PostflightReplay, PostflightStep, U16Access};
pub use preflight::{
    PreflightFieldBlock, PreflightHistory, PreflightInitialWrite, PreflightMemoryEvent,
    PreflightMemoryLog, PreflightOutput, PreflightProgramEvent,
};
#[cfg(feature = "rvr")]
pub use rvr::{
    PreflightEndpoint, PreflightExecution, PreflightInstance, PreflightLimits, PreflightTranscript,
};
pub use segment_scheduler::{
    drive_scheduled, Budget, ProvedBatch, ResourceProfile, ScheduledRun, SegmentDriver,
    SegmentNode, SegmentSchedulerConfig, SegmentSource, DEFAULT_PROVE_LOOKAHEAD, EXECUTE_GPU_BYTES,
    PROVE_MARGINAL_GPU_BYTES, SHARED_GPU_BASE_BYTES,
};
pub use state::*;
pub use vm::*;
