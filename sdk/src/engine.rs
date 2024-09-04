pub use afs_stark_backend::engine::StarkEngine;
use p3_uni_stark::StarkGenericConfig;

use crate::config::instrument::StarkHashStatistics;

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}
