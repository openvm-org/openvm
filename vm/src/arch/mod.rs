/// Instruction execution and machine chip traits and enum variants
mod chips;
/// Execution bus and interface
mod execution;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// Traits and builders to compose collections of chips into a virtual machine.
mod new_chipset;
/// Definitions of ProcessedInstruction types for use in integration API
mod processed_instructions; // rename to chipset once stable. Note: chipset is one word.

mod chip_set;
mod config;

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
pub use integration_api::*;
pub use processed_instructions::*;
pub use segment::*;
pub use vm::*;
