/// Instruction execution and machine chip traits and enum variants
mod chips;
/// Execution bus and interface
mod execution;
/// Traits and wrappers to faciliate VM chip integration
mod integration_api;

pub use axvm_instructions as instructions;

/// Testing framework
#[cfg(test)]
pub mod testing;

pub use chips::*;
pub use execution::*;
pub use integration_api::*;
