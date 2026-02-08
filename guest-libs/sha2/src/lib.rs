#![no_std]

#[cfg(not(any(target_os = "zkvm", feature = "import_sha2")))]
compile_error!("openvm-sha2 requires the 'import_sha2' feature to be enabled on non-zkvm targets");

#[cfg(feature = "import_sha2")]
pub use sha2::Digest;

#[cfg(not(target_os = "zkvm"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(not(target_os = "zkvm"))]
pub use host_impl::*;
#[cfg(target_os = "zkvm")]
pub use zkvm_impl::*;
