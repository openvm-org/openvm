//! OpenVM implementation of the eth-act zkVM Cryptographic Accelerators C Interface.
//!
//! This crate provides OpenVM's accelerated cryptographic operations: each function is exported as
//! a `#[no_mangle] extern "C"` symbol and delegates to the corresponding OpenVM accelerated guest
//! library.

#![no_std]

pub mod types;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod keccak256;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use keccak256::zkvm_keccak256;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod sha256;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use sha256::zkvm_sha256;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod secp256k1;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use secp256k1::{zkvm_secp256k1_ecrecover, zkvm_secp256k1_verify};
