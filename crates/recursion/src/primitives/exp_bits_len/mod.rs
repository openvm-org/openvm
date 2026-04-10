pub mod air;
pub mod trace;

pub use air::*;
pub use trace::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::ExpBitsLenGpuTraceGenerator;

#[cfg(feature = "cuda")]
pub type ExpBitsLenTraceGenerator = ExpBitsLenGpuTraceGenerator;

#[cfg(not(feature = "cuda"))]
pub type ExpBitsLenTraceGenerator = ExpBitsLenCpuTraceGenerator;
