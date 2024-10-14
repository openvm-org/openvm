use std::sync::Arc;

pub use afs_stark_backend::engine::StarkEngine;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    engine::VerificationData,
    rap::AnyRap,
    utils::AirInfo,
    verifier::VerificationError,
};
use p3_matrix::dense::DenseMatrix;
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

/// A struct that contains all the necessary data to:
/// - generate proving and verifying keys for AIRs,
/// - commit to trace matrices and generate STARK proofs
pub struct StarkForTest<SC: StarkGenericConfig> {
    pub air_infos: Vec<AirInfo<SC>>,
}

impl<SC: StarkGenericConfig> StarkForTest<SC> {
    pub fn run_test(
        self,
        engine: &impl StarkFriEngine<SC>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        engine.run_test(&self.air_infos)
    }
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
    fn new(fri_parameters: FriParameters) -> Self;
    fn fri_params(&self) -> FriParameters;
    fn run_test(
        &self,
        air_infos: &[AirInfo<SC>],
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let data = <Self as StarkEngine<_>>::run_test_impl(self, air_infos)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: self.fri_params(),
        })
    }
    fn run_test_fast(
        air_infos: Vec<AirInfo<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        engine.run_test(&air_infos)
    }
    fn run_simple_test_impl(
        &self,
        chips: Vec<Arc<dyn AnyRap<SC>>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        self.run_test(&AirInfo::multiple_simple(chips, traces, public_values))
    }
    fn run_simple_test_fast(
        chips: Vec<Arc<dyn AnyRap<SC>>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        StarkFriEngine::<_>::run_simple_test_impl(&engine, chips, traces, public_values)
    }
    fn run_simple_test_no_pis_fast(
        chips: Vec<Arc<dyn AnyRap<SC>>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let pis = vec![vec![]; chips.len()];
        <Self as StarkFriEngine<SC>>::run_simple_test_fast(chips, traces, pis)
    }
}
