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

/// A struct that contains all the necessary data to:
/// - generate proving and verifying keys for AIRs,
/// - commit to trace matrices and generate STARK proofs
pub struct StarkForTest<SC: StarkGenericConfig> {
    pub any_raps: Vec<Rc<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub pvs: Vec<Vec<Val<SC>>>,
}

impl<SC: StarkGenericConfig> StarkForTest<SC> {
    pub fn run_simple_test(
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
        let StarkForTest {
            any_raps,
            traces,
            pvs,
        } = self;
        let chips: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
        engine.run_simple_test_impl(&chips, traces, &pvs)
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
    fn run_simple_test_impl(
        &self,
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let data = <Self as StarkEngine<_>>::run_simple_test(self, chips, traces, public_values)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: self.fri_params(),
        })
    }
    fn run_simple_test(
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        StarkFriEngine::<_>::run_simple_test_impl(&engine, chips, traces, public_values)
    }
    fn run_simple_test_no_pis(
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        <Self as StarkFriEngine<SC>>::run_simple_test(chips, traces, &vec![vec![]; chips.len()])
    }
}
