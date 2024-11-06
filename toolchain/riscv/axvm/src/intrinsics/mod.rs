//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
pub use hash::*;

/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;
#[cfg(target_os = "zkvm")]
pub use io::*;

mod u256;
// pub use u256::*;

mod modular;
pub use modular::*;

mod fp2;
pub use fp2::*;

mod utils;
pub use utils::*;
