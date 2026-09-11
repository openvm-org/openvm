#![no_std]

#[cfg(feature = "import_sha2")]
pub use sha2::Digest;

#[cfg(all(
    not(any(openvm_intrinsics, target_os = "openvm")),
    feature = "import_sha2"
))]
mod host_impl;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod zkvm_impl;

#[cfg(all(
    not(any(openvm_intrinsics, target_os = "openvm")),
    feature = "import_sha2"
))]
pub use host_impl::*;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use zkvm_impl::*;
