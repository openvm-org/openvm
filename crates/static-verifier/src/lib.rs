#![forbid(unsafe_code)]

pub mod config;
pub mod gadgets;
pub mod stages;
mod utils;

pub use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

pub use config::{
    StaticVerifierShape, STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0,
    STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};
