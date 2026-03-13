// sha2 re-exports generic_array 0.x which is deprecated in favor of 1.x
#![allow(deprecated)]
#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

mod sha2_chips;
pub use sha2_chips::*;

mod extension;
pub use extension::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
