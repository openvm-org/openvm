#![no_std]

pub use sha2::Digest;

#[cfg(not(target_os = "zkvm"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(not(target_os = "zkvm"))]
pub use host_impl::*;
#[cfg(target_os = "zkvm")]
pub use zkvm_impl::*;
