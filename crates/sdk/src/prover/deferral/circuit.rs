use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_continuations::{
    circuit::deferral::{hook::DeferralIoCommit, DeferralCircuitPvs, DEF_CIRCUIT_PVS_AIR_ID},
    prover::{DeferralChildVkKind, DeferralCircuitProver},
    SC,
};
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_stark_backend::{keygen::types::MultiStarkProvingKey, proof::Proof, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};
use openvm_verify_stark_host::pvs::DagCommit;
use tracing::info_span;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::DeferralInnerGpuProver as DeferralInnerProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::DeferralInnerCpuProver as DeferralInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct SingleDefCircuitProver<'a> {
    pub def_circuit_prover: Box<dyn DeferralCircuitProver<SC> + 'a>,
    pub leaf_prover: DeferralInnerProver,
    pub internal_for_leaf_prover: DeferralInnerProver,
}

impl<'a> SingleDefCircuitProver<'a> {
    pub fn new<DP: DeferralCircuitProver<SC> + 'a>(
        def_circuit_prover: DP,
        leaf_params: SystemParams,
        internal_params: SystemParams,
    ) -> Self {
        let leaf_prover =
            DeferralInnerProver::new::<E>(def_circuit_prover.get_vk(), leaf_params, false);
        let internal_for_leaf_prover =
            DeferralInnerProver::new::<E>(leaf_prover.get_vk(), internal_params, false);
        Self {
            def_circuit_prover: Box::new(def_circuit_prover),
            leaf_prover,
            internal_for_leaf_prover,
        }
    }

    pub fn from_pks<DP: DeferralCircuitProver<SC> + 'a>(
        def_circuit_prover: DP,
        leaf_pk: Arc<MultiStarkProvingKey<SC>>,
        internal_for_leaf_pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self {
        let leaf_prover =
            DeferralInnerProver::from_pk::<E>(def_circuit_prover.get_vk(), leaf_pk, false);
        let internal_for_leaf_prover =
            DeferralInnerProver::from_pk::<E>(leaf_prover.get_vk(), internal_for_leaf_pk, false);
        Self {
            def_circuit_prover: Box::new(def_circuit_prover),
            leaf_prover,
            internal_for_leaf_prover,
        }
    }

    pub fn prove(&self, inputs: &[&[u8]]) -> Result<(Vec<Proof<SC>>, Vec<DeferralIoCommit<F>>)> {
        // Generate deferral circuit proofs
        let def_proofs = inputs
            .into_iter()
            .map(|input| self.def_circuit_prover.prove(input))
            .collect_vec();

        // Extract leaf IO commits from the deferral circuit proofs
        let leaf_children = def_proofs
            .iter()
            .map(|proof| {
                let pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
                    .as_slice()
                    .borrow();
                (pvs.input_commit, pvs.output_commit)
            })
            .collect();

        // Verify def-layer proofs and generate leaf-layer proofs
        let child_merkle_depth = (def_proofs.len() != 1).then_some(0);
        let leaf_proofs = info_span!("agg_layer", group = "def_leaf").in_scope(|| {
            def_proofs
                .chunks(2)
                .enumerate()
                .map(|(leaf_node_idx, proofs)| {
                    info_span!("single_leaf_agg", idx = leaf_node_idx).in_scope(|| {
                        self.leaf_prover.agg_prove::<E>(
                            proofs,
                            DeferralChildVkKind::DeferralCircuit,
                            child_merkle_depth,
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;

        // Verify leaf-layer proofs and generate internal-for-leaf-layer proofs
        let mut internal_node_idx = 0u32;
        let child_merkle_depth = (leaf_proofs.len() != 1).then_some(1);
        let internal_proofs =
            info_span!("agg_layer", group = "internal_for_leaf").in_scope(|| {
                leaf_proofs
                    .chunks(2)
                    .map(|proofs| {
                        let ret = info_span!("single_internal_agg", idx = internal_node_idx)
                            .in_scope(|| {
                                self.internal_for_leaf_prover.agg_prove::<E>(
                                    proofs,
                                    DeferralChildVkKind::DeferralAggregation,
                                    child_merkle_depth,
                                )
                            });
                        internal_node_idx += 1;
                        ret
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

        Ok((internal_proofs, leaf_children))
    }

    pub fn vk_commit(&self, internal_for_leaf_dag_commit: DagCommit<F>) -> Digest {
        let def_dag_commit = self.leaf_prover.get_dag_commit(false);
        let leaf_dag_commit = self.internal_for_leaf_prover.get_dag_commit(false);

        let vk_commit_components = vec![
            def_dag_commit.cached_commit,
            def_dag_commit.vk_pre_hash,
            leaf_dag_commit.cached_commit,
            leaf_dag_commit.vk_pre_hash,
            internal_for_leaf_dag_commit.cached_commit,
            internal_for_leaf_dag_commit.vk_pre_hash,
        ];
        poseidon2_hash_slice(&vk_commit_components.into_flattened()).0
    }
}
