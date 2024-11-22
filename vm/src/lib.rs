extern crate self as axvm_circuit;

pub use ax_circuit_derive as circuit_derive;
pub use axvm_circuit_derive as derive;

/// Traits and constructs for the axVM architecture.
pub mod arch;
/// Chips to support axVM intrinsic instructions.
pub mod intrinsics;
/// Chips to support axVM kernel instructions.
pub mod kernels;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
pub mod prover;
/// Chips to support RV32IM instructions.
pub mod rv32im;
/// System chips that are always required by the architecture.
/// (The [PhantomChip](system::phantom::PhantomChip) is not technically required for a functioning VM,
/// but there is almost always a need for it.)
pub mod system;
/// Utility functions and test utils
pub mod utils;
