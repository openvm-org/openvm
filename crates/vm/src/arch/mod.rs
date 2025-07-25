mod config;
/// Instruction execution traits and types.
/// Execution bus and interface.
pub mod execution;
/// Execution context types for different execution modes.
pub mod execution_mode;
/// Traits and builders to compose collections of chips into a virtual machine.
mod extensions;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// [RecordArena] trait definitions and implementations. Currently there are two concrete
/// implementations: [MatrixRecordArena] and [DenseRecordArena].
mod record_arena;
/// Runtime execution and segmentation
// TODO: rename this module
pub mod segment;
/// Top level [VmExecutor] and [VirtualMachine] constructor and API.
pub mod vm;

pub mod hasher;
/// Interpreter for VM execution
pub mod interpreter;
#[cfg(target_arch = "x86_64")]
pub mod tco;
/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use config::*;
pub use execution::*;
pub use execution_mode::{E1ExecutionCtx, E2ExecutionCtx};
pub use extensions::*;
pub use integration_api::*;
pub use openvm_instructions as instructions;
pub use record_arena::*;
pub use segment::*;
pub use vm::*;
