extern crate core;

use ax_stark_sdk::config::{
    baby_bear_poseidon2::BabyBearPoseidon2Config,
    baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
};
use axvm_recursion::types::InnerConfig;
use p3_baby_bear::BabyBear;

pub mod commit;
pub mod config;
// #[cfg(feature = "static-verifier")]
// pub mod static_verifier;

pub mod keygen;
pub mod prover;
pub mod verifier;

pub(crate) type SC = BabyBearPoseidon2Config;
pub(crate) type C = InnerConfig;
pub(crate) type F = BabyBear;
pub(crate) type OuterSC = BabyBearPoseidon2OuterConfig;
