use std::sync::Arc;

use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_continuations::{
    circuit::deferral::dummy::dummy_deferral_circuit_vk, prover::DeferralCircuitProver,
    CommitBytes, SC,
};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig},
    prover::{AggProver, DeferralPathProver, DeferralProver},
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

struct DeferralPathArtifacts {
    def_hook_commit: Digest,
    ir_vk: Arc<MultiStarkVerifyingKey<SC>>,
    ir_cached_commit: CommitBytes,
    agg_prover: Arc<AggProver>,
}

impl DeferralPathProver {
    /// Derives the deferral path's fixed-point artifacts with a cheap dummy deferral circuit.
    ///
    /// These artifacts depend only on the aggregation and hook params, not on the concrete
    /// deferral circuit whose proofs are later fed into the path.
    fn fixed_point_artifacts(
        agg_params: &AggregationSystemParams,
        hook_params: &SystemParams,
    ) -> DeferralPathArtifacts {
        let dummy = DummyDefCircuitProver {
            vk: dummy_deferral_circuit_vk::<E>(agg_params.internal.clone()),
        };

        let agg_config = AggregationConfig {
            params: agg_params.clone(),
        };
        let deferral_prover = DeferralProver::new(dummy, agg_config.clone(), hook_params.clone());

        let deferral_tree_config = AggregationTreeConfig {
            num_children_leaf: 2,
            num_children_internal: 2,
        };
        let agg_prover = Arc::new(AggProver::new(
            deferral_prover.def_hook_prover.get_vk(),
            agg_config,
            deferral_tree_config,
            Some(deferral_prover.def_hook_prover.get_cached_commit()),
        ));

        // The deferral-path aggregation tree's internal-recursive vk is a universal copy of the VM
        // internal-recursive vk that a verify-stark circuit verifies.
        let ir_vk = agg_prover.internal_recursive_prover.get_vk();
        let ir_cached_commit = agg_prover
            .internal_recursive_prover
            .get_self_vk_pcs_data()
            .expect("internal-recursive prover must expose its self vk pcs data")
            .commitment
            .into();
        let path_prover = Self::new(Arc::new(deferral_prover), agg_prover.clone());

        DeferralPathArtifacts {
            def_hook_commit: path_prover.def_hook_commit(),
            ir_vk,
            ir_cached_commit,
            agg_prover,
        }
    }

    /// Builds a [`DeferralPathProver`] backed by the verify-stark circuit, configured so an SDK
    /// with the given params can recursively verify the VM STARK proofs it produces, including its
    /// own deferral-carrying proofs.
    ///
    /// The deferral-enabled internal-recursive vk and the self-referential `def_hook_commit` are
    /// derived internally from a dummy deferral circuit.
    pub fn verify_stark(
        agg_params: &AggregationSystemParams,
        hook_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self {
        let DeferralPathArtifacts {
            def_hook_commit,
            ir_vk,
            ir_cached_commit,
            agg_prover,
        } = Self::fixed_point_artifacts(agg_params, &hook_params);

        let deferred_verify_prover = VerifyProver::new::<E>(
            ir_vk,
            ir_cached_commit,
            agg_params.internal.clone(),
            memory_dimensions,
            num_user_pvs,
            Some(def_hook_commit),
            0,
        );
        let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);

        let agg_config = AggregationConfig {
            params: agg_params.clone(),
        };
        let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);
        Self::new(Arc::new(deferral_prover), agg_prover)
    }
}

/// A dummy [`DeferralCircuitProver`] that only exposes a trivial verifying key. It exists solely to
/// seed the deferral aggregation chain when deriving the deferral path fixed point; its `prove`
/// method is never called.
struct DummyDefCircuitProver {
    vk: Arc<MultiStarkVerifyingKey<SC>>,
}

impl DeferralCircuitProver<SC> for DummyDefCircuitProver {
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    fn prove(&self, _input_bytes: &[u8]) -> Proof<SC> {
        unreachable!("DummyDefCircuitProver is only used to derive deferral path artifacts")
    }

    fn get_def_idx(&self) -> usize {
        0
    }
}
