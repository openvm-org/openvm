//! We need to export a custom getrandom implementation just to get crates that import getrandom to
//! compile.
use getrandom::{register_custom_getrandom, Error};

/// This is a getrandom handler for the zkvm. It's intended to hook into a
/// getrandom crate or a dependent of the getrandom crate used by the guest code.
#[cfg(feature = "getrandom")]
pub fn zkvm_getrandom(dest: &mut [u8]) -> Result<(), Error> {
    todo!()
    // Randomness would come from the host
}

#[cfg(not(feature = "getrandom"))]
pub fn zkvm_getrandom(dest: &mut [u8]) -> Result<(), Error> {
    panic!("getrandom is not enabled in the current build");
}

register_custom_getrandom!(zkvm_getrandom);
