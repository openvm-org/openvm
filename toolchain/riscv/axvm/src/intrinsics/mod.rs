//! Functions that call custom instructions that use axVM intrinsic instructions.

mod hash;
/// Library functions for user input/output.
pub mod io;

pub use hash::*;
pub use io::*;

/// Exit the program with exit code 0.
pub fn exit() {
    axvm_platform::rust_rt::terminate::<0>();
}

/// Exit the program with exit code 1.
pub fn panic() {
    axvm_platform::rust_rt::terminate::<1>();
}
