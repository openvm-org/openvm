use std::sync::Arc;

use eyre::Result;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    prover::CommittedTraceData,
    StarkEngine, SystemParams,
};
use tracing::info_span;
use verify_stark::NonRootStarkProof;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use continuations_v2::aggregation::CompressionGpuProver as CompressionInnerProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use continuations_v2::aggregation::CompressionCpuProver as CompressionInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct CompressionProver(pub CompressionInnerProver);

impl CompressionProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<crate::SC>>,
        internal_recursive_vk_pcs_data: CommittedTraceData<<E as StarkEngine>::PB>,
        system_params: SystemParams,
    ) -> Self {
        let inner = CompressionInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            system_params,
        );
        Self(inner)
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<crate::SC>>,
        internal_recursive_vk_pcs_data: CommittedTraceData<<E as StarkEngine>::PB>,
        pk: Arc<MultiStarkProvingKey<crate::SC>>,
    ) -> Self {
        let inner = CompressionInnerProver::from_pk::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            pk,
        );
        Self(inner)
    }

    pub fn prove(&self, mut proof: NonRootStarkProof) -> Result<NonRootStarkProof> {
        proof.inner = info_span!("agg_layer", group = format!("compression")).in_scope(|| {
            info_span!("compression").in_scope(|| self.0.compress_prove::<E>(proof.inner))
        })?;
        Ok(proof)
    }
}
