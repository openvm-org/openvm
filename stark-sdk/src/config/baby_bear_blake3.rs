use p3_blake3::Blake3;

use super::{
    baby_bear_bytehash::{
        self, config_from_byte_hash, BabyBearByteHashConfig, BabyBearByteHashEngine,
    },
    FriParameters,
};
use crate::config::baby_bear_bytehash::BabyBearByteHashEngineWithDefaultHash;

pub type BabyBearBlake3Config = BabyBearByteHashConfig<Blake3>;
pub type BabyBearBlake3Engine = BabyBearByteHashEngine<Blake3>;

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> BabyBearBlake3Engine {
    baby_bear_bytehash::default_engine(Blake3)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config() -> BabyBearBlake3Config {
    let fri_params = FriParameters::standard_fast();
    config_from_byte_hash(Blake3, fri_params)
}

impl BabyBearByteHashEngineWithDefaultHash<Blake3> for BabyBearBlake3Engine {
    fn default_hash() -> Blake3 {
        Blake3
    }
}
