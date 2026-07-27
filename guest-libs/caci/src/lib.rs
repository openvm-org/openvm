//! OpenVM implementation of the eth-act zkVM Cryptographic Accelerators C Interface.
//!
//! This crate provides OpenVM's accelerated cryptographic operations: each function is exported as
//! a `#[no_mangle] extern "C"` symbol and delegates to the corresponding OpenVM accelerated guest
//! library.

#![no_std]

pub mod types;
