#![no_std]

#[cfg(feature = "import_sha2")]
pub use sha2::Digest;

#[cfg(all(not(openvm_intrinsics), feature = "import_sha2"))]
mod host_impl;
#[cfg(openvm_intrinsics)]
mod zkvm_impl;

#[cfg(all(not(openvm_intrinsics), feature = "import_sha2"))]
pub use host_impl::*;
#[cfg(openvm_intrinsics)]
pub use zkvm_impl::*;
