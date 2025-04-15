mod config;
/// Instruction execution traits and types.
/// Execution bus and interface.
mod execution;
/// Module for controlling VM execution flow, including segmentation and instruction execution
pub mod execution_control;
/// Traits and builders to compose collections of chips into a virtual machine.
mod extensions;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// Runtime execution and segmentation
pub mod segment;
/// Strategy for determining when to segment VM execution
pub mod segmentation_strategy;
/// Top level [VirtualMachine] constructor and API.
pub mod vm;

pub use openvm_instructions as instructions;

pub mod hasher;
/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use config::*;
pub use execution::*;
pub use extensions::*;
pub use integration_api::*;
pub use segment::*;
pub use segmentation_strategy::*;
pub use vm::*;
