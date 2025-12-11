use std::sync::Arc;

use continuations_v2::aggregation::AggregationProver;
use eyre::Result;
use openvm_circuit::arch::ContinuationVmProof;
use stark_backend_v2::{SC, keygen::types::MultiStarkVerifyingKeyV2};
use tracing::info_span;
use verify_stark::NonRootStarkProof;

use crate::{
    config::{
        AggregationConfig, AggregationTreeConfig, MAX_NUM_CHILDREN_INTERNAL, MAX_NUM_CHILDREN_LEAF,
    },
    keygen::AggProvingKey,
};

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
    pub leaf_prover: NonRootAggregationProver<MAX_NUM_CHILDREN_LEAF>,
    pub internal_for_leaf_prover: NonRootAggregationProver<MAX_NUM_CHILDREN_INTERNAL>,
    pub internal_recursive_prover: NonRootAggregationProver<MAX_NUM_CHILDREN_INTERNAL>,
    pub agg_tree_config: AggregationTreeConfig,
}

impl AggProver {
    pub fn new(
        app_vk: Arc<MultiStarkVerifyingKeyV2>,
        agg_config: AggregationConfig,
        agg_tree_config: AggregationTreeConfig,
    ) -> Self {
        assert!(agg_tree_config.num_children_leaf <= MAX_NUM_CHILDREN_LEAF);
        assert!(agg_tree_config.num_children_internal <= MAX_NUM_CHILDREN_INTERNAL);
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

    pub fn from_pk(
        app_vk: Arc<MultiStarkVerifyingKeyV2>,
        agg_pk: AggProvingKey,
        agg_tree_config: AggregationTreeConfig,
    ) -> Self {
        let leaf_prover = NonRootAggregationProver::from_pk::<E>(app_vk, agg_pk.leaf_pk, false);
        let internal_for_leaf_prover = NonRootAggregationProver::from_pk::<E>(
            leaf_prover.get_vk(),
            agg_pk.internal_for_leaf_pk,
            false,
        );
        let internal_recursive_prover = NonRootAggregationProver::from_pk::<E>(
            internal_for_leaf_prover.get_vk(),
            agg_pk.internal_recursive_pk,
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
        let leaf_proofs = info_span!("agg_layer", group = "leaf").in_scope(|| {
            continuation_proof
                .per_segment
                .chunks(self.agg_tree_config.num_children_leaf)
                .enumerate()
                .map(|(leaf_node_idx, proofs)| {
                    info_span!("single_leaf_agg", idx = leaf_node_idx).in_scope(|| {
                        self.leaf_prover.agg_prove::<E>(
                            proofs,
                            Some(continuation_proof.user_public_values.public_values_commit),
                            false,
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;

        // Verify leaf-layer proofs and generate internal-for-leaf-layer proofs
        let mut internal_node_idx = -1;
        let mut internal_proofs =
            info_span!("agg_layer", group = "internal_for_leaf").in_scope(|| {
                leaf_proofs
                    .chunks(self.agg_tree_config.num_children_internal)
                    .map(|proofs| {
                        internal_node_idx += 1;
                        info_span!("single_internal_agg", idx = internal_node_idx).in_scope(|| {
                            self.internal_for_leaf_prover
                                .agg_prove::<E>(proofs, None, false)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

        // Verify internal-for-leaf-layer proofs and generate internal-recursive-layer proofs
        internal_proofs =
            info_span!("agg_layer", group = "internal_recursive.0").in_scope(|| {
                internal_proofs
                    .chunks(self.agg_tree_config.num_children_internal)
                    .map(|proofs| {
                        internal_node_idx += 1;
                        info_span!("single_internal_agg", idx = internal_node_idx).in_scope(|| {
                            self.internal_recursive_prover
                                .agg_prove::<E>(proofs, None, false)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

        // Recursively verify internal-layer proofs until only 1 remains
        let mut internal_recursive_layer = 1;
        while internal_proofs.len() > 1 {
            internal_proofs = info_span!(
                "agg_layer",
                group = format!("internal_recursive.{internal_recursive_layer}")
            )
            .in_scope(|| {
                internal_proofs
                    .chunks(self.agg_tree_config.num_children_internal)
                    .map(|proofs| {
                        internal_node_idx += 1;
                        info_span!("single_internal_agg", idx = internal_node_idx).in_scope(|| {
                            self.internal_recursive_prover
                                .agg_prove::<E>(proofs, None, true)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            internal_recursive_layer += 1;
        }

        Ok(NonRootStarkProof {
            inner: internal_proofs[0].clone(),
            user_pvs_proof: continuation_proof.user_public_values,
        })
    }
}
