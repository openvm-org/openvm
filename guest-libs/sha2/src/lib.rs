#![no_std]

#[cfg(feature = "import_sha2")]
pub use sha2::Digest;

#[cfg(all(not(target_os = "zkvm"), feature = "import_sha2"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(all(not(target_os = "zkvm"), feature = "import_sha2"))]
pub use host_impl::*;
#[cfg(target_os = "zkvm")]
pub use zkvm_impl::*;
