#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

pub mod call;
pub mod count;
pub mod output;
pub mod setup;
pub(crate) mod utils;

mod extension;
pub use extension::*;
