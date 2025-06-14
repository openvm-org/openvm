use openvm_native_recursion::types::InnerConfig;
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig,
    },
    p3_baby_bear::BabyBear,
};

#[cfg(feature = "static-verifier")]
pub mod static_verifier;
pub mod verifier;

pub type SC = BabyBearPoseidon2Config;
pub type C = InnerConfig;
pub type F = BabyBear;
pub type RootSC = BabyBearPoseidon2RootConfig;
