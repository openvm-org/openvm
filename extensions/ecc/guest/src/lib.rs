#![no_std]

extern crate alloc;
extern crate self as axvm_ecc_guest;

pub use axvm_algebra_guest as algebra;
#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;
mod ecdsa;
pub use ecdsa::*;

/// Weierstrass curve traits
pub mod sw;

/// Types for Secp256k1 curve with intrinsic functions. Implements traits necessary for ECDSA.
#[cfg(feature = "k256")]
pub mod k256;
