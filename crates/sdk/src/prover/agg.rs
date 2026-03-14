use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::ContinuationVmProof;
use openvm_continuations::{circuit::inner::ProofsType, prover::ChildVkKind};
use openvm_cuda_backend::prelude::Digest;
use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_compress_with_capacity, F};
use openvm_verify_stark_host::{pvs::DeferralPvs, NonRootStarkProof};
use tracing::info_span;

use crate::{
    config::{
        AggregationConfig, AggregationTreeConfig, MAX_NUM_CHILDREN_INTERNAL, MAX_NUM_CHILDREN_LEAF,
    },
    keygen::AggProvingKey,
    prover::deferral::DeferralProof,
    SC,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::InnerGpuProver as InnerAggregationProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::InnerCpuProver as InnerAggregationProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct AggProver {
    pub leaf_prover: InnerAggregationProver<MAX_NUM_CHILDREN_LEAF>,
    pub internal_for_leaf_prover: InnerAggregationProver<MAX_NUM_CHILDREN_INTERNAL>,
    pub internal_recursive_prover: InnerAggregationProver<MAX_NUM_CHILDREN_INTERNAL>,
    pub agg_tree_config: AggregationTreeConfig,
}

pub struct InternalLayerMetadata {
    pub internal_recursive_layer: u32,
    pub internal_node_idx: u32,
}

impl AggProver {
    pub fn new(
        app_or_def_vk: Arc<MultiStarkVerifyingKey<SC>>,
        agg_config: AggregationConfig,
        agg_tree_config: AggregationTreeConfig,
        def_hook_commit: Option<Digest>,
    ) -> Self {
        assert!(agg_tree_config.num_children_leaf <= MAX_NUM_CHILDREN_LEAF);
        assert!(agg_tree_config.num_children_internal <= MAX_NUM_CHILDREN_INTERNAL);
        let leaf_prover = InnerAggregationProver::new::<E>(
            app_or_def_vk,
            agg_config.params.leaf.clone(),
            false,
            def_hook_commit,
        );
        let internal_for_leaf_prover = InnerAggregationProver::new::<E>(
            leaf_prover.get_vk(),
            agg_config.params.internal.clone(),
            false,
            def_hook_commit,
        );
        let internal_recursive_prover = InnerAggregationProver::new::<E>(
            internal_for_leaf_prover.get_vk(),
            agg_config.params.internal.clone(),
            true,
            def_hook_commit,
        );
        Self {
            leaf_prover,
            internal_for_leaf_prover,
            internal_recursive_prover,
            agg_tree_config,
        }
    }

    pub fn from_pk(
        app_or_def_vk: Arc<MultiStarkVerifyingKey<SC>>,
        agg_pk: AggProvingKey,
        agg_tree_config: AggregationTreeConfig,
        def_hook_commit: Option<Digest>,
    ) -> Self {
        let leaf_prover =
            InnerAggregationProver::from_pk::<E>(app_or_def_vk, agg_pk.leaf_pk, false, None);
        let internal_for_leaf_prover = InnerAggregationProver::from_pk::<E>(
            leaf_prover.get_vk(),
            agg_pk.internal_for_leaf_pk,
            false,
            def_hook_commit,
        );
        let internal_recursive_prover = InnerAggregationProver::from_pk::<E>(
            internal_for_leaf_prover.get_vk(),
            agg_pk.internal_recursive_pk,
            true,
            def_hook_commit,
        );
        Self {
            leaf_prover,
            internal_for_leaf_prover,
            internal_recursive_prover,
            agg_tree_config,
        }
    }

    pub fn prove_vm(
        &self,
        continuation_proof: ContinuationVmProof<SC>,
    ) -> Result<(NonRootStarkProof, InternalLayerMetadata)> {
        // Verify app-layer proofs and generate leaf-layer proofs
        let leaf_proofs = info_span!("agg_layer", group = "leaf").in_scope(|| {
            continuation_proof
                .per_segment
                .chunks(self.agg_tree_config.num_children_leaf)
                .enumerate()
                .map(|(leaf_node_idx, proofs)| {
                    info_span!("single_leaf_agg", idx = leaf_node_idx).in_scope(|| {
                        self.leaf_prover
                            .agg_prove_no_def::<E>(proofs, ChildVkKind::App)
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
                                .agg_prove_no_def::<E>(proofs, ChildVkKind::Standard)
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
                                .agg_prove_no_def::<E>(proofs, ChildVkKind::Standard)
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
                                .agg_prove_no_def::<E>(proofs, ChildVkKind::RecursiveSelf)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            internal_recursive_layer += 1;
        }

        Ok((
            NonRootStarkProof {
                inner: internal_proofs.pop().unwrap(),
                user_pvs_proof: continuation_proof.user_public_values,
            },
            InternalLayerMetadata {
                internal_recursive_layer: internal_recursive_layer as u32,
                internal_node_idx: internal_node_idx as u32,
            },
        ))
    }

    pub fn prove_def(&self, input: Vec<DeferralProof>) -> Result<DeferralProof> {
        assert!(!input.is_empty());

        // Leaf round: hook-level → leaf-level
        let mut proofs = info_span!("agg_layer", group = "def_leaf")
            .in_scope(|| reduce_def_round(input, ChildVkKind::App, &self.leaf_prover))?;

        // Internal-for-leaf round: leaf-level → i4l-level
        proofs = info_span!("agg_layer", group = "def_internal_for_leaf").in_scope(|| {
            reduce_def_round(
                proofs,
                ChildVkKind::Standard,
                &self.internal_for_leaf_prover,
            )
        })?;

        // Internal-recursive round 0: i4l-level → ir-level
        proofs = info_span!("agg_layer", group = "def_internal_recursive.0").in_scope(|| {
            reduce_def_round(
                proofs,
                ChildVkKind::Standard,
                &self.internal_recursive_prover,
            )
        })?;

        // Internal-recursive rounds: ir-level → ir-level until single proof remains
        let mut layer = 1;
        while proofs.len() > 1 {
            proofs = info_span!(
                "agg_layer",
                group = format!("def_internal_recursive.{layer}")
            )
            .in_scope(|| {
                reduce_def_round(
                    proofs,
                    ChildVkKind::RecursiveSelf,
                    &self.internal_recursive_prover,
                )
            })?;
            layer += 1;
        }

        Ok(proofs.pop().unwrap())
    }

    pub fn prove_mixed(
        &self,
        mut vm_proof: NonRootStarkProof,
        def_proof: DeferralProof,
        metadata: &mut InternalLayerMetadata,
    ) -> Result<NonRootStarkProof> {
        let DeferralProof::Present(def_inner) = def_proof else {
            return Ok(vm_proof);
        };

        vm_proof.inner = info_span!(
            "agg_layer",
            group = format!("internal_recursive.{}", metadata.internal_recursive_layer)
        )
        .in_scope(|| {
            metadata.internal_recursive_layer += 1;
            info_span!("single_internal_agg", idx = metadata.internal_node_idx).in_scope(|| {
                metadata.internal_node_idx += 1;
                self.internal_recursive_prover.agg_prove::<E>(
                    &[vm_proof.inner, def_inner],
                    ChildVkKind::RecursiveSelf,
                    ProofsType::Mix,
                    None,
                )
            })
        })?;

        Ok(vm_proof)
    }

    pub fn wrap_proof(
        &self,
        mut proof: NonRootStarkProof,
        metadata: &mut InternalLayerMetadata,
        proofs_type: ProofsType,
    ) -> Result<NonRootStarkProof> {
        proof.inner = info_span!(
            "agg_layer",
            group = format!("internal_recursive.{}", metadata.internal_recursive_layer)
        )
        .in_scope(|| {
            metadata.internal_recursive_layer += 1;
            info_span!("single_internal_agg", idx = metadata.internal_node_idx).in_scope(|| {
                metadata.internal_node_idx += 1;
                self.internal_recursive_prover.agg_prove::<E>(
                    &[proof.inner],
                    ChildVkKind::RecursiveSelf,
                    proofs_type,
                    None,
                )
            })
        })?;
        Ok(proof)
    }
}

fn reduce_def_round<const N: usize>(
    proofs: Vec<DeferralProof>,
    kind: ChildVkKind,
    prover: &InnerAggregationProver<N>,
) -> Result<Vec<DeferralProof>> {
    let mut next = Vec::with_capacity(proofs.len().div_ceil(2));
    let mut iter = proofs.into_iter();
    while let Some(a) = iter.next() {
        match iter.next() {
            Some(b) => {
                let combined = match (a, b) {
                    (DeferralProof::Present(p0), DeferralProof::Present(p1)) => {
                        DeferralProof::Present(prover.agg_prove::<E>(
                            &[p0, p1],
                            kind,
                            ProofsType::Deferral,
                            None,
                        )?)
                    }
                    (DeferralProof::Present(p), DeferralProof::Absent(pvs)) => {
                        // Absent is the right child (present is left, is_right = false)
                        DeferralProof::Present(prover.agg_prove::<E>(
                            &[p],
                            kind,
                            ProofsType::Deferral,
                            Some((pvs, false)),
                        )?)
                    }
                    (DeferralProof::Absent(pvs), DeferralProof::Present(p)) => {
                        // Absent is the left child (present is right, is_right = true)
                        DeferralProof::Present(prover.agg_prove::<E>(
                            &[p],
                            kind,
                            ProofsType::Deferral,
                            Some((pvs, true)),
                        )?)
                    }
                    (DeferralProof::Absent(pvs0), DeferralProof::Absent(pvs1)) => {
                        DeferralProof::Absent(DeferralPvs {
                            initial_acc_hash: poseidon2_compress_with_capacity(
                                pvs0.initial_acc_hash,
                                pvs1.initial_acc_hash,
                            )
                            .0,
                            final_acc_hash: poseidon2_compress_with_capacity(
                                pvs0.final_acc_hash,
                                pvs1.final_acc_hash,
                            )
                            .0,
                            depth: pvs0.depth + F::ONE,
                        })
                    }
                };
                next.push(combined);
            }
            None => {
                // Trailing singleton: wrap Present, pass Absent unchanged
                let out =
                    match a {
                        DeferralProof::Present(p) => DeferralProof::Present(
                            prover.agg_prove::<E>(&[p], kind, ProofsType::Deferral, None)?,
                        ),
                        absent => absent,
                    };
                next.push(out);
            }
        }
    }
    Ok(next)
}
