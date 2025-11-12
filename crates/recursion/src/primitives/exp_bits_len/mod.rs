pub mod air;
pub mod trace;

pub use air::*;
pub use trace::*;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

// TODO(ayush): feature gate this to be CPU or GPU version
pub type ExpBitsLenTraceGenerator = ExpBitsLenCpuTraceGenerator;
