#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

pub mod adapters;
pub mod call;
pub mod canonicity;
pub mod count;
pub mod output;
pub mod poseidon2;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_abi;
pub(crate) mod utils;

mod def_fn;
pub use def_fn::*;
mod extension;
pub use extension::*;
