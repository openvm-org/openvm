mod challenger;
mod commit;
mod folder;
pub mod fri;
pub mod hints;
mod poseidon2;
pub mod stark;
#[cfg(test)]
mod tests;
pub mod types;
mod utils;
mod digest;

/// Digest size in the outer config.
const OUTER_DIGEST_SIZE: usize = 1;
