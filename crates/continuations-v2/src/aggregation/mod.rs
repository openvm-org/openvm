use std::sync::Arc;

#[cfg(feature = "cuda")]
use cuda_backend_v2::GpuBackendV2;
use eyre::Result;
use itertools::Itertools;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use recursion_circuit::system::{AggregationSubCircuit, VerifierSubCircuit};
use stark_backend_v2::{
    DIGEST_SIZE, F, StarkWhirEngine,
    keygen::types::MultiStarkVerifyingKeyV2,
    proof::Proof,
    prover::{CpuBackendV2, ProverBackendV2, ProvingContextV2},
};

use crate::public_values::{
    NonRootTraceGen, receiver::UserPvsReceiverAir, verifier::VerifierPvsAir,
};

mod nonroot;
mod root;
mod utils;

pub use nonroot::*;
pub use root::*;
pub use utils::*;

pub const MAX_NUM_PROOFS: usize = 4;

pub trait AggregationProver<PB: ProverBackendV2> {
    // Verifying key used to verify the result of agg_prove
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2>;
    // Commit of verifier circuit's cached trace
    fn get_commit(&self) -> PB::Commitment;
    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> ProvingContextV2<PB>;
    fn agg_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Result<Proof>;
}

pub type NonRootCpuProver =
    NonRootAggregationProver<CpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGen>;
#[cfg(feature = "cuda")]
pub type NonRootGpuProver =
    NonRootAggregationProver<GpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGen>;

#[derive(derive_new::new, Clone)]
pub struct AggregationCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> AggregationCircuit<S> {
    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let public_values_bus = self.verifier_circuit.public_values_bus();
        [Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus: self.verifier_circuit.cached_commit_bus(),
        }) as AirRef<BabyBearPoseidon2Config>]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain([
            Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<BabyBearPoseidon2Config>
        ])
        .collect_vec()
    }
}
