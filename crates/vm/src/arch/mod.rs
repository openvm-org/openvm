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
pub(crate) mod preflight;
#[cfg(feature = "rvr")]
pub mod rvr;
/// VM state definitions
mod state;
/// Top level [VmExecutor] and [VirtualMachine] constructor and API.
pub mod vm;

pub mod hasher;
/// Interpreter implementations for pure, metered, and preflight execution.
pub mod interpreter;
/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use config::*;
pub use execution::*;
pub use execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait};
pub use extensions::*;
pub use hint_stream::HintStream;
pub use integration_api::*;
pub use interpreter::{InterpretedInstance, PreflightInterpretedInstance};
pub use openvm_circuit_derive::create_handler;
pub use openvm_instructions as instructions;
pub use postflight::{
    fill_trace_rows, Postflight, PostflightError, PostflightProgramIndex, PostflightReplay,
    PostflightStep, POSTFLIGHT_PREDECESSOR_INDEX_LIMIT,
};
#[rustfmt::skip]
pub use postflight::{U8Access, U16Access, Field32Access};
pub use preflight::{
    PreflightFieldBlock, PreflightHistory, PreflightInitialWrite, PreflightMemoryEvent,
    PreflightMemoryLog, PreflightOutput, PreflightProgramEvent,
};
pub use state::*;
pub use vm::*;
