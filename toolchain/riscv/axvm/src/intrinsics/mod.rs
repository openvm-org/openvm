//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
pub use hash::*;

/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;
#[cfg(target_os = "zkvm")]
pub use io::*;

mod u256;
pub use u256::*;

mod i256;
pub use i256::*;

mod utils;
#[allow(unused)]
pub use utils::*;
