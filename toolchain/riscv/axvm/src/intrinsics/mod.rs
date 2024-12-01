//! Functions that call custom instructions that use axVM intrinsic instructions.

/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;
#[cfg(target_os = "zkvm")]
pub use io::*;

mod utils;
#[allow(unused)]
pub use utils::*;
