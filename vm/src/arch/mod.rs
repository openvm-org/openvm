mod config;
/// Instruction execution traits and types.
/// Execution bus and interface.
mod execution;
/// Traits and builders to compose collections of chips into a virtual machine.
mod extensions;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// Runtime execution and segmentation
pub mod new_segment; // replace segment::* once stable
/// Top level [VirtualMachine] constructor and API.
pub mod new_vm; // replace vm::* once stable

// to be deleted once extensions is stable
mod chips;
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
