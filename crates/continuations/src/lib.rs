use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config;

pub mod bn254;
pub mod circuit;
pub mod prover;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

pub type SC = recursion_circuit::prelude::SC;
pub type RootSC = BabyBearBn254Poseidon2Config;
