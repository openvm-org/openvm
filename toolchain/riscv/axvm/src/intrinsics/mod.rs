//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;

pub use hash::*;
#[cfg(target_os = "zkvm")]
pub use io::*;

mod utils;
#[allow(unused_imports)] // This is used in the rust-v programs
use utils::*;
