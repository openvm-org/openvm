use serde::{Deserialize, Serialize};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

pub mod baby_bear_blake3;
pub mod baby_bear_bytehash;
pub mod baby_bear_keccak;
pub mod baby_bear_poseidon2;
pub mod fri_params;
pub mod instrument;

pub fn setup_tracing() {
    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum EngineType {
    BabyBearBlake3,
    BabyBearKeccak,
    BabyBearPoseidon2,
}
