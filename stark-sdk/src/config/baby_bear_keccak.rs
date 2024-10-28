use p3_keccak::Keccak256Hash;

use super::{
    baby_bear_bytehash::{
        self, config_from_byte_hash, BabyBearByteHashConfig, BabyBearByteHashEngine,
    },
    FriParameters,
};
use crate::config::baby_bear_bytehash::BabyBearByteHashEngineWithDefaultHash;

pub type BabyBearKeccakConfig = BabyBearByteHashConfig<Keccak256Hash>;
pub type BabyBearKeccakEngine = BabyBearByteHashEngine<Keccak256Hash>;

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine(pcs_log_degree: usize) -> BabyBearKeccakEngine {
    baby_bear_bytehash::default_engine(pcs_log_degree, Keccak256Hash)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(pcs_log_degree: usize) -> BabyBearKeccakConfig {
    let fri_params = FriParameters::standard_fast();
    config_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params)
}

impl BabyBearByteHashEngineWithDefaultHash<Keccak256Hash> for BabyBearKeccakEngine {
    fn default_hash() -> Keccak256Hash {
        Keccak256Hash
    }
}
