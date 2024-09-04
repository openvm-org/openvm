use derive_more::Display;
use serde::{Deserialize, Serialize};
use tracing::Level;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

pub mod baby_bear_blake3;
pub mod baby_bear_bytehash;
pub mod baby_bear_keccak;
pub mod baby_bear_poseidon2;
pub mod baby_bear_poseidon2_outer;
pub mod fri_params;
pub mod goldilocks_poseidon;
pub mod instrument;

pub fn setup_tracing() {
    setup_tracing_with_log_level(Level::INFO);
}

pub fn setup_tracing_with_log_level(level: Level) {
    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FriParameters {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

#[derive(Clone, Copy, Default, Display, Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum EngineType {
    #[default]
    BabyBearPoseidon2,
    BabyBearBlake3,
    BabyBearKeccak,
    GoldilocksPoseidon,
}
