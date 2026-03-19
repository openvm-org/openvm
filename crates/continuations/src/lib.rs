pub mod circuit;
pub mod prover;
pub mod utils;

mod commit_bytes;
pub use commit_bytes::*;

#[cfg(test)]
mod tests;

pub type SC = openvm_recursion_circuit::prelude::SC;
#[cfg(feature = "root-prover")]
pub type RootSC = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config;
