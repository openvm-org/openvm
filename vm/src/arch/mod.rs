/// Instruction execution and machine chip traits and enum variants
mod chips;
/// Execution bus and interface
mod execution;
/// Traits and wrappers to facilitate VM chip integration
mod integration_api;
/// Definitions of ProcessedInstruction types for use in integration API
mod processed_instructions;

pub use axvm_instructions as instructions;

/// Testing framework
#[cfg(test)]
pub mod testing;

pub use chips::*;
pub use execution::*;
pub use integration_api::*;
pub use processed_instructions::*;
