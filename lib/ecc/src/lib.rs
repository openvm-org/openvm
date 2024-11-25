#![no_std]

extern crate alloc;

pub use axvm_algebra as algebra;
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

/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
pub mod halo2curves_shims;
/// Traits for optimal Ate pairing check using intrinsic functions.
pub mod pairing;
/// Weierstrass curve traits
pub mod sw;

/// Types for BLS12-381 curve with intrinsic functions.
pub mod bls12_381;
/// Types for BN254 curve with intrinsic functions.
pub mod bn254;
