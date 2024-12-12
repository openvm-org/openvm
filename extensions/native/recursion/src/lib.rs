pub mod challenger;
mod commit;
pub mod config;
pub mod digest;
mod folder;
pub mod fri;
mod helper;
pub mod hints;
mod outer_poseidon2;
pub mod stark;
pub mod types;
pub mod utils;
pub mod vars;
mod view;
pub mod witness;

#[cfg(feature = "static-verifier")]
pub mod halo2;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing_utils;
#[cfg(test)]
mod tests;

/// Digest size in the outer config.
const OUTER_DIGEST_SIZE: usize = 1;
