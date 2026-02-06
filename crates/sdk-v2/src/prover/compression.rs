use std::sync::Arc;

use eyre::Result;
use stark_backend_v2::{
    StarkEngineV2, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    prover::CommittedTraceDataV2,
};
use tracing::info_span;
use verify_stark::NonRootStarkProof;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use continuations_v2::aggregation::CompressionGpuProver as CompressionInnerProver;
        type E = cuda_backend_v2::BabyBearPoseidon2GpuEngineV2;
    } else {
        use continuations_v2::aggregation::CompressionCpuProver as CompressionInnerProver;
        type E = stark_backend_v2::BabyBearPoseidon2CpuEngineV2;
    }
}

pub struct CompressionProver(pub CompressionInnerProver);

impl CompressionProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKeyV2>,
        internal_recursive_vk_pcs_data: CommittedTraceDataV2<<E as StarkEngineV2>::PB>,
        system_params: SystemParams,
    ) -> Self {
        let inner = CompressionInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            system_params,
        );
        Self { 0: inner }
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKeyV2>,
        internal_recursive_vk_pcs_data: CommittedTraceDataV2<<E as StarkEngineV2>::PB>,
        pk: Arc<MultiStarkProvingKeyV2>,
    ) -> Self {
        let inner = CompressionInnerProver::from_pk::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            pk,
        );
        Self { 0: inner }
    }

    pub fn prove(&self, mut proof: NonRootStarkProof) -> Result<NonRootStarkProof> {
        proof.inner = info_span!("agg_layer", group = format!("compression")).in_scope(|| {
            info_span!("compression").in_scope(|| self.0.compress_prove::<E>(proof.inner))
        })?;
        Ok(proof)
    }
}
