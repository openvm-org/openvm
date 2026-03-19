//! Static verifier circuit for OpenVM root STARK proof.
//! The verifier circuit is implemented using Halo2 via the `halo2-base` eDSL.
//!
//! Static means that the circuit hard codes the following and does not allow them to vary as part
//! of the input:
//! - The child verifying key, including all system parameters
//! - The trace heights of the root proof (the static verifier circuit's input) are **fixed**. The
//!   heights of each AIR are fixed. Consequently the permutation order of AIRs sorted by height is
//!   fixed.
//! - The trace heights of the root proof are all nonzero. In other words no AIR in the child
//!   verifying key is optional.
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
    StaticVerifierShape, STATIC_VERIFIER_LOOKUP_ADVICE_COLS, STATIC_VERIFIER_NUM_ADVICE_COLS,
};
pub use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
pub use keygen::StaticVerifierProvingKey;
pub use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{EF as RootEF, F as RootF};
pub use prover::{
    Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierCircuit,
    StaticVerifierInput, StaticVerifierProof,
};
