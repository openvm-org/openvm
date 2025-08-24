#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
extern crate self as openvm_circuit;

pub use openvm_circuit_derive as derive;
pub use openvm_circuit_primitives_derive as circuit_derive;
#[cfg(all(feature = "test-utils", feature = "cuda"))]
pub use openvm_cuda_backend;
#[cfg(feature = "test-utils")]
pub use openvm_stark_sdk;

/// Traits and constructs for the OpenVM architecture.
pub mod arch;
/// Instrumentation metrics for performance analysis and debugging
#[cfg(feature = "metrics")]
pub mod metrics;
/// System chips that are always required by the architecture.
/// (The [PhantomChip](system::phantom::PhantomChip) is not technically required for a functioning
/// VM, but there is almost always a need for it.)
pub mod system;
/// Utility functions and test utils
pub mod utils;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_abi;
