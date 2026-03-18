#![forbid(unsafe_code)]

pub mod config;
pub mod field;
pub mod hash;
pub mod keygen;
pub mod prover;
pub mod stages;
pub mod transcript;
mod utils;

pub use config::{
    StaticVerifierShape, STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0,
    STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};
pub use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
pub use keygen::StaticVerifierProvingKey;
pub use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{EF as ChildEF, F as ChildF};
pub use prover::{
    Halo2Params, Halo2Prover, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierInput,
    StaticVerifierProof,
};
