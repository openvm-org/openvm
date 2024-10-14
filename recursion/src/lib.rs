mod challenger;
mod commit;
pub mod config;
mod digest;
mod folder;
pub mod fri;
mod helper;
pub mod hints;
mod outer_poseidon2;
pub mod stark;
pub mod testing_utils;
pub mod types;
mod utils;
pub mod vars;
mod view;
pub mod witness;

#[cfg(feature = "static-verifier")]
pub mod halo2;

#[cfg(test)]
mod tests;

/// Digest size in the outer config.
const OUTER_DIGEST_SIZE: usize = 1;
