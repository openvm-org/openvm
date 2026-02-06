#![no_std]
extern crate self as openvm_ecc_guest;

pub mod edwards;

pub use openvm_weierstrass_guest as weierstrass;
