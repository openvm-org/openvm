#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

mod auipc;
mod base_alu;
mod base_alu_w;
mod branch_eq;
mod branch_lt;
mod divrem;
mod divrem_w;
mod extension;
mod jal_lui;
mod jalr;
mod less_than;
mod mul;
mod mul_w;
mod mulh;
mod shift;
mod shift_w;
#[cfg(test)]
mod test_utils;

pub use auipc::*;
pub use base_alu::*;
pub use base_alu_w::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use divrem::*;
pub use divrem_w::*;
pub use extension::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use mul::*;
pub use mul_w::*;
pub use mulh::*;
pub use shift::*;
pub use shift_w::*;
