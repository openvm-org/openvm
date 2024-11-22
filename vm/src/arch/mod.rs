/// Instruction execution and machine chip traits and enum variants
mod chips;
mod config;
/// Execution bus and interface
mod execution;
/// Traits and builders to compose collections of chips into a virtual machine.
mod extensions;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
mod new_segment;

// delete once extensions is stable
mod chip_set;
// delete once new_segment is stable
#[macro_use]
mod segment;

mod vm;

pub use axvm_instructions as instructions;

pub mod hasher;
/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use chip_set::*;
pub use chips::*;
pub use config::*;
pub use execution::*;
pub use extensions::*;
pub use integration_api::*;
pub use segment::*;
pub use vm::*;
