mod challenger;
mod commit;
pub mod config;
mod digest;
mod folder;
pub mod fri;
#[cfg(feature = "static-verifier")]
pub mod halo2;
pub mod hints;
mod outer_poseidon2;
pub mod stark;
pub mod testing_utils;
#[cfg(test)]
mod tests;
pub mod types;
mod utils;
pub mod v2;
pub mod witness;

/// Digest size in the outer config.
const OUTER_DIGEST_SIZE: usize = 1;
