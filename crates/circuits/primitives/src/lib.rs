//! This crate contains a collection of primitives for use when building circuits.
//! The primitives are separated into two types: standalone
//! [Air](openvm_stark_backend::p3_air::Air)s and [SubAir]s.
//!
//! The following modules contain standalone [Air](openvm_stark_backend::p3_air::Air)s:
//! - [range]
//! - [range_gate]
//! - [range_tuple]
//! - [var_range]
//! - [xor]
//!
//! The following modules contain [SubAir]s:
//! - [assert_less_than]
//! - [bigint]
//! - [bitwise_op_lookup]
//! - [encoder]
//! - [is_equal]
//! - [is_equal_array]
//! - [is_less_than]
//! - [is_less_than_array]
//! - [is_zero]

/// Derive macros
pub use openvm_circuit_primitives_derive::*;

pub mod assert_less_than;
pub mod bigint;
pub mod bitwise_op_lookup;
pub mod encoder;
pub mod is_equal;
pub mod is_equal_array;
pub mod is_less_than;
pub mod is_less_than_array;
pub mod is_zero;
pub mod range;
pub mod range_gate;
pub mod range_tuple;
pub mod utils;
pub mod var_range;
pub mod xor;

mod sub_air;
pub use sub_air::*;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_abi;
