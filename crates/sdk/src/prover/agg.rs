use std::sync::Arc;

use openvm_native_circuit::NativeConfig;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    openvm_stark_backend::prover::types::Proof,
};
use tracing::info_span;

// #[cfg(feature = "bench-metrics")]
use super::vm::types::VmProvingKey;
use crate::{
    keygen::AggStarkProvingKey,
    prover::{
        vm::{local::VmLocalProver, ContinuationVmProof, SingleSegmentVmProver},
        RootVerifierLocalProver,
    },
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
    NonRootCommittedExe, RootSC, F, SC,
};

const DEFAULT_NUM_CHILDREN_LEAF: usize = 1;
const DEFAULT_NUM_CHILDREN_INTERNAL: usize = 2;
const DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS: usize = 4;

pub struct AggStarkProver {
    leaf_prover: LeafProver,
    internal_prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    root_prover: RootVerifierLocalProver,

    pub num_children_internal: usize,
    pub max_internal_wrapper_layers: usize,
}
pub struct LeafProver {
    prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    /// Each leaf proof aggregations `<= num_children` App VM proofs
    pub num_children: usize,
}

impl AggStarkProver {
    pub fn new(
        agg_stark_pk: AggStarkProvingKey,
        leaf_committed_exe: Arc<NonRootCommittedExe>,
    ) -> Self {
        let leaf_prover = LeafProver::new(agg_stark_pk.leaf_vm_pk, leaf_committed_exe);
        let internal_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            agg_stark_pk.internal_vm_pk,
            agg_stark_pk.internal_committed_exe,
        );
        let root_prover = RootVerifierLocalProver::new(agg_stark_pk.root_verifier_pk);
        Self {
            leaf_prover,
            internal_prover,
            root_prover,
            num_children_internal: DEFAULT_NUM_CHILDREN_INTERNAL,
            max_internal_wrapper_layers: DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS,
        }
    }

    pub fn with_num_children_leaf(mut self, num_children_leaf: usize) -> Self {
        self.leaf_prover.num_children = num_children_leaf;
        self
    }

    pub fn with_num_children_internal(mut self, num_children_internal: usize) -> Self {
        self.num_children_internal = num_children_internal;
        self
    }

    pub fn with_max_internal_wrapper_layers(mut self, max_internal_wrapper_layers: usize) -> Self {
        self.max_internal_wrapper_layers = max_internal_wrapper_layers;
        self
    }

    /// Generate a proof to aggregate app proofs.
    pub fn generate_agg_proof(&self, app_proofs: ContinuationVmProof<SC>) -> Proof<RootSC> {
        let leaf_proofs = self.leaf_prover.generate_proof(&app_proofs);
        let public_values = app_proofs.user_public_values.public_values;
        let internal_proof = self.generate_internal_proof_impl(leaf_proofs, &public_values);
        self.generate_root_proof_impl(RootVmVerifierInput {
            proofs: vec![internal_proof],
            public_values,
        })
    }

    fn generate_internal_proof_impl(
        &self,
        leaf_proofs: Vec<Proof<SC>>,
        public_values: &[F],
    ) -> Proof<SC> {
        let mut internal_node_idx = -1;
        let mut internal_node_height = 0;
        let mut proofs = leaf_proofs;
        let mut wrapper_layers = 0;
        loop {
            // TODO: what's a good test case for the wrapping logic?
            if proofs.len() == 1 {
                // TODO: record execution time as a part of root verifier execution time.
                let actual_air_heights =
                    self.root_prover
                        .execute_for_air_heights(RootVmVerifierInput {
                            proofs: vec![proofs[0].clone()],
                            public_values: public_values.to_vec(),
                        });
                // Root verifier can handle the internal proof. We can stop here.
                if heights_le(
                    &actual_air_heights,
                    &self.root_prover.root_verifier_pk.air_heights,
                ) {
                    break;
                }
                if wrapper_layers >= self.max_internal_wrapper_layers {
                    panic!("The heights of the root verifier still exceed the required heights after {} wrapper layers", self.max_internal_wrapper_layers);
                }
                wrapper_layers += 1;
            }
            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                self.internal_prover
                    .committed_exe
                    .get_program_commit()
                    .into(),
                &proofs,
                self.num_children_internal,
            );
            proofs = info_span!(
                "agg_layer",
                group = format!("internal.{internal_node_height}")
            )
            .in_scope(|| {
                #[cfg(feature = "bench-metrics")]
                {
                    metrics::counter!("fri.log_blowup")
                        .absolute(self.internal_prover.fri_params().log_blowup as u64);
                    metrics::counter!("num_children").absolute(self.num_children_internal as u64);
                }
                internal_inputs
                    .into_iter()
                    .map(|input| {
                        internal_node_idx += 1;
                        info_span!("single_internal_agg", idx = internal_node_idx,).in_scope(|| {
                            SingleSegmentVmProver::prove(&self.internal_prover, input.write())
                        })
                    })
                    .collect()
            });
            internal_node_height += 1;
        }
        proofs.pop().unwrap()
    }

    fn generate_root_proof_impl(&self, root_input: RootVmVerifierInput<SC>) -> Proof<RootSC> {
        info_span!("agg_layer", group = "root", idx = 0).in_scope(|| {
            let input = root_input.write();
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.root_prover.fri_params().log_blowup as u64);
            SingleSegmentVmProver::prove(&self.root_prover, input)
        })
    }
}

impl LeafProver {
    pub fn new(
        leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
        leaf_committed_exe: Arc<NonRootCommittedExe>,
    ) -> Self {
        let prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            leaf_vm_pk,
            leaf_committed_exe,
        );
        Self {
            prover,
            num_children: DEFAULT_NUM_CHILDREN_LEAF,
        }
    }
    pub fn with_num_children(mut self, num_children_leaf: usize) -> Self {
        self.num_children = num_children_leaf;
        self
    }
    pub fn generate_proof(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
        info_span!("agg_layer", group = "leaf").in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            {
                metrics::counter!("fri.log_blowup").absolute(self.fri_params().log_blowup as u64);
                metrics::counter!("num_children").absolute(self.num_children as u64);
            }
            let leaf_inputs =
                LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children);
            tracing::info!("num_leaf_proofs={}", leaf_inputs.len());
            leaf_inputs
                .into_iter()
                .enumerate()
                .map(|(leaf_node_idx, input)| {
                    info_span!("single_leaf_agg", idx = leaf_node_idx).in_scope(|| {
                        SingleSegmentVmProver::prove(&self.prover, input.write_to_stream())
                    })
                })
                .collect::<Vec<_>>()
        })
    }
    #[allow(dead_code)]
    pub(crate) fn fri_params(&self) -> &FriParameters {
        &self.prover.pk.fri_params
    }
}

fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}
