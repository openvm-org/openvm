use std::sync::Arc;

use continuations_v2::aggregation::AggregationProver;
use eyre::Result;
use openvm_circuit::arch::ContinuationVmProof;
use stark_backend_v2::{SC, keygen::types::MultiStarkVerifyingKeyV2};
use verify_stark::NonRootStarkProof;

use crate::config::{AggregationConfig, AggregationTreeConfig};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use continuations_v2::aggregation::NonRootGpuProver as NonRootAggregationProver;
        type E = cuda_backend_v2::BabyBearPoseidon2GpuEngineV2;
    } else {
        use continuations_v2::aggregation::NonRootCpuProver as NonRootAggregationProver;
        type E = stark_backend_v2::BabyBearPoseidon2CpuEngineV2;
    }
}

pub struct AggProver {
    pub leaf_prover: NonRootAggregationProver,
    pub internal_for_leaf_prover: NonRootAggregationProver,
    pub internal_recursive_prover: NonRootAggregationProver,
    pub agg_tree_config: AggregationTreeConfig,
}

impl AggProver {
    pub fn new(
        app_vk: Arc<MultiStarkVerifyingKeyV2>,
        agg_config: AggregationConfig,
        agg_tree_config: AggregationTreeConfig,
    ) -> Self {
        let leaf_prover = NonRootAggregationProver::new::<E>(app_vk, agg_config.params.leaf, false);
        let internal_for_leaf_prover = NonRootAggregationProver::new::<E>(
            leaf_prover.get_vk(),
            agg_config.params.internal,
            false,
        );
        let internal_recursive_prover = NonRootAggregationProver::new::<E>(
            internal_for_leaf_prover.get_vk(),
            agg_config.params.internal,
            true,
        );
        Self {
            leaf_prover,
            internal_for_leaf_prover,
            internal_recursive_prover,
            agg_tree_config,
        }
    }

    pub fn prove(&self, continuation_proof: ContinuationVmProof<SC>) -> Result<NonRootStarkProof> {
        // Verify app-layer proofs and generate leaf-layer proofs
        let leaf_proofs = continuation_proof
            .per_segment
            .chunks(self.agg_tree_config.num_children_leaf)
            .map(|proofs| {
                self.leaf_prover.agg_prove::<E>(
                    proofs,
                    Some(continuation_proof.user_public_values.public_values_commit),
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Verify leaf-layer proofs and generate internal-for-leaf-layer proofs
        let mut internal_proofs = leaf_proofs
            .chunks(self.agg_tree_config.num_children_internal)
            .map(|proofs| {
                self.internal_for_leaf_prover
                    .agg_prove::<E>(proofs, None, false)
            })
            .collect::<Result<Vec<_>>>()?;

        // Verify internal-for-leaf-layer proofs and generate internal-recursive-layer proofs
        internal_proofs = internal_proofs
            .chunks(self.agg_tree_config.num_children_internal)
            .map(|proofs| {
                self.internal_recursive_prover
                    .agg_prove::<E>(proofs, None, false)
            })
            .collect::<Result<Vec<_>>>()?;

        // Recursively verify internal-layer proofs until only 1 remains
        while internal_proofs.len() > 1 {
            internal_proofs = internal_proofs
                .chunks(self.agg_tree_config.num_children_internal)
                .map(|proofs| {
                    self.internal_recursive_prover
                        .agg_prove::<E>(proofs, None, true)
                })
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(NonRootStarkProof {
            inner: internal_proofs[0].clone(),
            user_pvs_proof: continuation_proof.user_public_values,
        })
    }
}
