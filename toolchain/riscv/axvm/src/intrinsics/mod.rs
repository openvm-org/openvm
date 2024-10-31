//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
/// Library functions for user input/output.
pub mod io;

pub use hash::*;
pub use io::*;
