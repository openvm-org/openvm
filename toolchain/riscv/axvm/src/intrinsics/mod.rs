//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
mod int256;
/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;

pub use hash::*;
pub use int256::*;
#[cfg(target_os = "zkvm")]
pub use io::*;
