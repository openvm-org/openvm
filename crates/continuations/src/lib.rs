use openvm_native_recursion::types::InnerConfig;
use openvm_stark_sdk::{
    config::{
        koala_bear_poseidon2::KoalaBearPoseidon2Config,
        koala_bear_poseidon2_root::KoalaBearPoseidon2RootConfig,
    },
    p3_koala_bear::KoalaBear,
};

pub mod static_verifier;
pub mod verifier;

pub type SC = KoalaBearPoseidon2Config;
pub type C = InnerConfig;
pub type F = KoalaBear;
pub type RootSC = KoalaBearPoseidon2RootConfig;
