//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
/// Utilities to work with the hint stream.
pub mod io;

pub use hash::*;
pub use io::*;
