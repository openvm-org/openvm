// Note that we put the ed25519 module here to avoid a cyclic dependency between openvm_ecc_guest
// and openvm_edwards_guest. The root cause of the cycle is that te_declare! uses
// openvm_ecc_guest.
#[cfg(feature = "ed25519")]
pub mod ed25519;

pub use openvm_edwards_guest::*;
