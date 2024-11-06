use std::sync::Arc;

pub use ax_stark_backend::engine::StarkEngine;
use ax_stark_backend::{
    config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
    engine::VerificationData,
    prover::types::AirProofInput,
    rap::AnyRap,
    verifier::VerificationError,
};
use p3_matrix::dense::DenseMatrix;
use tracing::Level;

use crate::config::{instrument::StarkHashStatistics, setup_tracing_with_log_level, FriParameters};

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

/// All necessary data to verify a Stark proof.
pub struct VerificationDataWithFriParams<SC: StarkGenericConfig> {
    pub data: VerificationData<SC>,
    pub fri_params: FriParameters,
}

/// `stark-backend::prover::types::ProofInput` without specifying AIR IDs.
pub struct ProofInputForTest<SC: StarkGenericConfig> {
    pub per_air: Vec<AirProofInput<SC>>,
}

impl<SC: StarkGenericConfig> ProofInputForTest<SC> {
    pub fn run_test(
        self,
        engine: &impl StarkFriEngine<SC>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        engine.run_test(self.per_air)
    }
}

/// Stark engine using Fri.
pub trait StarkFriEngine<SC: StarkGenericConfig>: StarkEngine<SC> + Sized
where
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
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        setup_tracing_with_log_level(Level::WARN);
        let data = <Self as StarkEngine<_>>::run_test_impl(self, air_proof_inputs)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: self.fri_params(),
        })
    }
    fn run_test_fast(
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        let engine = Self::new(FriParameters::standard_fast());
        engine.run_test(air_proof_inputs)
    }
    fn run_simple_test_impl(
        &self,
        chips: Vec<Arc<dyn AnyRap<SC>>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        self.run_test(AirProofInput::multiple_simple(chips, traces, public_values))
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
