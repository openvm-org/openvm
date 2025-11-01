mod core;
pub use core::*;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

// TODO[stephen]: feature gate this to be CPU or GPU version
pub type PowerCheckerTraceGenerator<const BASE: usize, const N: usize> =
    PowerCheckerCpuTraceGenerator<BASE, N>;
