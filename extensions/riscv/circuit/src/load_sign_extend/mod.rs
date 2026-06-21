pub(crate) mod aligned;
mod byte;
mod halfword;
mod word;

pub use byte::*;
pub use halfword::*;
pub use word::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
