use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_continuations::{CommitBytes, RootSC, SC};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::ProvingContext,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;
use openvm_verify_stark_host::NonRootStarkProof;
use tracing::info_span;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::RootCpuProver as RootInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    } else {
        use openvm_continuations::prover::RootCpuProver as RootInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

pub struct RootProver(pub RootInnerProver);

impl RootProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_dag_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_vk_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_dag_commit,
            system_params,
            memory_dimensions,
            num_user_pvs,
            def_hook_vk_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_dag_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<RootSC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_vk_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::from_pk::<E>(
            internal_recursive_vk,
            internal_recursive_dag_commit,
            pk,
            memory_dimensions,
            num_user_pvs,
            def_hook_vk_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn generate_proving_ctx(
        &self,
        input: NonRootStarkProof,
    ) -> Option<ProvingContext<<E as StarkEngine>::PB>> {
        let ctx = info_span!("tracegen_attempt", group = format!("root")).in_scope(|| {
            self.0.generate_proving_ctx(
                input.inner,
                &input.user_pvs_proof,
                input.deferral_merkle_proofs.as_ref(),
            )
        });
        ctx
    }

    pub fn prove_from_ctx(
        &self,
        ctx: ProvingContext<<E as StarkEngine>::PB>,
    ) -> Result<Proof<RootSC>> {
        let proof = info_span!("agg_layer", group = format!("root"))
            .in_scope(|| info_span!("root").in_scope(|| self.0.root_prove_from_ctx::<E>(ctx)))?;
        Ok(proof)
    }
}
