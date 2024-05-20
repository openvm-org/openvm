#![feature(trait_upcasting)]
use afs_stark_backend::{
    keygen::types::SymbolicRap, prover::types::ProverRap, verifier::types::VerifierRap,
};
use p3_uni_stark::StarkGenericConfig;

pub mod config;
pub mod interaction;
pub mod utils;

pub trait ProverVerifierRap<SC: StarkGenericConfig>:
    ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>
{
}
impl<SC: StarkGenericConfig, RAP: ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>>
    ProverVerifierRap<SC> for RAP
{
}
