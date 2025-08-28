mod config;
/// Instruction execution traits and types.
/// Execution bus and interface.
pub mod execution;
/// Execution context types for different execution modes.
pub mod execution_mode;
mod extensions;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// [RecordArena] trait definitions and implementations. Currently there are two concrete
/// implementations: [MatrixRecordArena] and [DenseRecordArena].
mod record_arena;
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
pub use integration_api::*;
pub use interpreter::InterpretedInstance;
pub use openvm_circuit_derive::create_tco_handler;
pub use openvm_instructions as instructions;
pub use record_arena::*;
pub use state::*;
pub use vm::*;
