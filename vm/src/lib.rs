pub use ax_circuit_derive as circuit_derive;
pub use axvm_circuit_derive as derive;

/// Traits and constructs for the axVM architecture.
pub mod arch;
/// Common chips that are not specific to a particular context.
pub mod common;
/// Chips to support axVM intrinsic instructions.
pub mod intrinsics;
/// Chips to support axVM kernel instructions.
pub mod kernels;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
/// Chips to support RV32IM instructions.
pub mod rv32im;
/// System chips that are always required by the architecture.
pub mod system;

#[cfg(feature = "sdk")]
pub mod sdk;
mod utils;

// To be deleted:
pub mod old;
