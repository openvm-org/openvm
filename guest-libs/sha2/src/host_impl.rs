// On a host execution environment, the zkvm impl's input buffering is not necessary, and we can
// use the sha2 crate directly.
pub use sha2::{Sha256, Sha384, Sha512};
