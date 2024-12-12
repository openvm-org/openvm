extern crate self as axvm_circuit;

pub use ax_circuit_derive as circuit_derive;
#[cfg(feature = "test-utils")]
pub use ax_stark_sdk;
pub use axvm_circuit_derive as derive;

/// Traits and constructs for the axVM architecture.
pub mod arch;
/// Instrumentation metrics for performance analysis and debugging
pub mod metrics;
/// System chips that are always required by the architecture.
/// (The [PhantomChip](system::phantom::PhantomChip) is not technically required for a functioning VM,
/// but there is almost always a need for it.)
pub mod system;
/// Utility functions and test utils
pub mod utils;
