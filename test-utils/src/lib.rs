#![feature(trait_upcasting)]
use afs_stark_backend::keygen::types::SymbolicRap;
use afs_stark_backend::prover::types::ProverRap;
use afs_stark_backend::verifier::types::VerifierRap;
use p3_uni_stark::StarkGenericConfig;

pub mod cached_lookup;
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

pub fn get_conditional_fib_number(sels: &[bool]) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = a + b;
            a = b;
            b = c;
        }
    }
    b
}
