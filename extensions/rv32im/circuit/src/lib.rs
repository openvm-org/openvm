pub mod adapters;

pub mod auipc;
pub mod base_alu;
pub mod branch_eq;
pub mod branch_lt;
pub mod divrem;
pub mod hintstore;
pub mod jal_lui;
pub mod jalr;
pub mod less_than;
pub mod load_sign_extend;
pub mod loadstore;
pub mod mul;
pub mod mulh;
pub mod shift;

pub use auipc::*;
pub use base_alu::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use divrem::*;
pub use hintstore::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
pub use mul::*;
pub use mulh::*;
pub use shift::*;

mod extension;
pub use extension::*;

#[cfg(any(test, feature = "test-utils"))]
mod test_utils;
