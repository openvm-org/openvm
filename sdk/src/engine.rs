use std::rc::Rc;

pub use afs_stark_backend::engine::StarkEngine;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    engine::VerificationData,
    rap::AnyRap,
    verifier::VerificationError,
};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::config::{instrument::StarkHashStatistics, FriParameters};

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

/// All necessary data to verify a Stark proof.
pub struct VerificationDataWithFriParams<SC: StarkGenericConfig> {
    pub data: VerificationData<SC>,
    pub fri_params: FriParameters,
}

/// A struct that contains all the necessary data to build a verifier for a Stark.
pub struct StarkForTest<SC: StarkGenericConfig> {
    pub any_raps: Vec<Rc<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub pvs: Vec<Vec<Val<SC>>>,
}

/// Stark engine using Fri.
pub trait StarkFriEngine<SC: StarkGenericConfig>: StarkEngine<SC> + Sized
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    fn default_engine(max_log_degree: usize) -> Self;
    fn fri_params(&self) -> FriParameters;
    fn run_simple_test(
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let engine = Self::default_engine(27);
        let data = engine.run_simple_test(chips, traces, public_values)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: engine.fri_params(),
        })
    }
    fn run_simple_test_no_pis(
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        <Self as StarkFriEngine<SC>>::run_simple_test(chips, traces, &vec![vec![]; chips.len()])
    }
}
