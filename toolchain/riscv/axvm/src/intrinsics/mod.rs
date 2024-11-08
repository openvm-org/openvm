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

// TODO[jpw]: move this to axvm-ecc; currently axvm_ecc::sw is calling moduli_setup! which breaks things
mod modular;
pub use modular::*;

mod utils;
pub use utils::*;
