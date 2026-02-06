use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use stark_backend_v2::{
    StarkEngineV2, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    proof::Proof,
    prover::CommittedTraceDataV2,
};
use tracing::info_span;
use verify_stark::NonRootStarkProof;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use continuations_v2::aggregation::RootGpuProver as RootInnerProver;
        type E = cuda_backend_v2::BabyBearPoseidon2GpuEngineV2;
    } else {
        use continuations_v2::aggregation::RootCpuProver as RootInnerProver;
        type E = stark_backend_v2::BabyBearPoseidon2CpuEngineV2;
    }
}

pub struct RootProver(pub RootInnerProver);

impl RootProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKeyV2>,
        internal_recursive_vk_pcs_data: CommittedTraceDataV2<<E as StarkEngineV2>::PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self {
        let inner = RootInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            system_params,
            memory_dimensions,
            num_user_pvs,
        );
        Self { 0: inner }
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKeyV2>,
        internal_recursive_vk_pcs_data: CommittedTraceDataV2<<E as StarkEngineV2>::PB>,
        pk: Arc<MultiStarkProvingKeyV2>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self {
        let inner = RootInnerProver::from_pk(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            pk,
            memory_dimensions,
            num_user_pvs,
        );
        Self { 0: inner }
    }

    pub fn prove(&self, input: NonRootStarkProof) -> Result<Proof> {
        let proof = info_span!("agg_layer", group = format!("root")).in_scope(|| {
            info_span!("root")
                .in_scope(|| self.0.root_prove::<E>(input.inner, &input.user_pvs_proof))
        })?;
        Ok(proof)
    }
}
