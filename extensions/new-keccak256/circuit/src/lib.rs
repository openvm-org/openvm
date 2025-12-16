#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

mod keccakf;
mod xorin;

#[cfg(feature = "cuda")]
mod cuda;

mod extension;
pub use extension::*;
