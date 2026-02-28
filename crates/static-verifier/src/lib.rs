#![forbid(unsafe_code)]

pub mod circuit;
pub mod config;
pub mod gadgets;
pub mod stages;
mod utils;

pub use circuit::StaticVerifierCircuit;
pub use config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
    StaticVerifierShape,
};
