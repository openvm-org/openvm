use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_continuations::{prover::engine_device_ctx, CommitBytes, RootSC, SC};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::ProvingContext,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;
use openvm_verify_stark_host::VmStarkProof;
use tracing::info_span;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::RootGpuProver as RootInnerProver;
        type E = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::RootCpuProver as RootInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

pub struct RootProver(pub RootInnerProver);

impl RootProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_vk_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_vk_commit,
            system_params,
            memory_dimensions,
            num_user_pvs,
            def_hook_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_vk_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<RootSC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::from_pk::<E>(
            internal_recursive_vk,
            internal_recursive_vk_commit,
            pk,
            memory_dimensions,
            num_user_pvs,
            def_hook_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn create_engine(&self) -> E {
        self.0.create_engine::<E>()
    }

    pub fn generate_proving_ctx(
        &self,
        input: VmStarkProof,
        engine: &E,
    ) -> Option<ProvingContext<<E as StarkEngine>::PB>> {
        let ctx = info_span!("tracegen_attempt", group = format!("root")).in_scope(|| {
            self.0.generate_proving_ctx(
                input.inner,
                &input.user_pvs_proof,
                input.deferral_merkle_proofs.as_ref(),
                engine_device_ctx(engine),
            )
        });
        ctx
    }

    pub fn prove_from_ctx(
        &self,
        ctx: ProvingContext<<E as StarkEngine>::PB>,
        engine: &E,
    ) -> Result<Proof<RootSC>> {
        let proof = info_span!("agg_layer", group = format!("root")).in_scope(|| {
            info_span!("root").in_scope(|| self.0.root_prove_from_ctx::<E>(ctx, engine))
        })?;
        Ok(proof)
    }

    pub fn prove(
        &self,
        mut stark_proof: VmStarkProof,
        engine: &E,
        max_retries: usize,
        mut wrap: impl FnMut(VmStarkProof) -> Result<VmStarkProof>,
    ) -> Result<Proof<RootSC>> {
        let mut attempt = 0usize;
        let ctx = loop {
            if let Some(ctx) = self.generate_proving_ctx(stark_proof.clone(), engine) {
                break ctx;
            }
            if attempt >= max_retries {
                return Err(eyre::eyre!(
                    "root tracegen returned None after {max_retries} retries"
                ));
            }
            stark_proof = wrap(stark_proof)?;
            attempt += 1;
        };

        // Internal sanity (SDK tests only): a successful tracegen must land at
        // exactly the root verifier's fixed, expected trace heights.
        #[cfg(test)]
        for ((air_idx, air_ctx), expected_height) in ctx
            .per_trace
            .iter()
            .zip(self.0.get_trace_heights().unwrap())
        {
            assert_eq!(
                air_ctx.height(),
                expected_height,
                "height mismatch at {air_idx}"
            );
        }

        self.prove_from_ctx(ctx, engine)
    }
}
