pub mod bn254;
pub mod circuit;
pub mod prover;

#[cfg(test)]
mod tests;

pub type SC = recursion_circuit::prelude::SC;
