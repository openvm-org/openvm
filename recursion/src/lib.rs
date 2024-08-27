mod challenger;
mod commit;
pub mod config;
mod digest;
mod folder;
pub mod fri;
pub mod halo2;
pub mod hints;
mod outer_poseidon2;
pub mod stark;
#[cfg(test)]
mod tests;
pub mod types;
mod utils;
mod witness;

/// Digest size in the outer config.
const OUTER_DIGEST_SIZE: usize = 1;
