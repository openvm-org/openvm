use std::sync::Arc;

use openvm_circuit::arch::ContinuationVmProof;
use openvm_continuations::verifier::{
    internal::types::{InternalVmVerifierInput, VmStarkProof},
    leaf::types::LeafVmVerifierInput,
    root::types::RootVmVerifierInput,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::{engine::StarkFriEngine, openvm_stark_backend::proof::Proof};
use tracing::info_span;

use crate::{
    config::AggregationTreeConfig,
    keygen::AggStarkProvingKey,
    prover::{
        vm::{local::VmLocalProver, SingleSegmentVmProver},
        RootVerifierLocalProver,
    },
    NonRootCommittedExe, RootSC, F, SC,
};

pub struct AggStarkProver<E: StarkFriEngine<SC>> {
    leaf_prover: VmLocalProver<SC, NativeConfig, E>,
    leaf_controller: LeafProvingController,

    internal_prover: VmLocalProver<SC, NativeConfig, E>,
    root_prover: RootVerifierLocalProver,

    pub num_children_internal: usize,
    pub max_internal_wrapper_layers: usize,
}

pub struct LeafProvingController {
    /// Each leaf proof aggregations `<= num_children` App VM proofs
    pub num_children: usize,
}

impl<E: StarkFriEngine<SC>> AggStarkProver<E> {
    pub fn new(
        agg_stark_pk: AggStarkProvingKey,
        leaf_committed_exe: Arc<NonRootCommittedExe>,
        tree_config: AggregationTreeConfig,
    ) -> Self {
        let leaf_prover =
            VmLocalProver::<SC, NativeConfig, E>::new(agg_stark_pk.leaf_vm_pk, leaf_committed_exe);
        let leaf_controller = LeafProvingController {
            num_children: tree_config.num_children_leaf,
        };
        let internal_prover = VmLocalProver::<SC, NativeConfig, E>::new(
            agg_stark_pk.internal_vm_pk,
            agg_stark_pk.internal_committed_exe,
        );
        let root_prover = RootVerifierLocalProver::new(agg_stark_pk.root_verifier_pk);
        Self {
            leaf_prover,
            leaf_controller,
            internal_prover,
            root_prover,
            num_children_internal: tree_config.num_children_internal,
            max_internal_wrapper_layers: tree_config.max_internal_wrapper_layers,
        }
    }

    pub fn with_num_children_leaf(mut self, num_children_leaf: usize) -> Self {
        self.leaf_controller.num_children = num_children_leaf;
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

    /// Generate the root proof for outer recursion.
    pub fn generate_root_proof(&self, app_proofs: ContinuationVmProof<SC>) -> Proof<RootSC> {
        let root_verifier_input = self.generate_root_verifier_input(app_proofs);
        self.generate_root_proof_impl(root_verifier_input)
    }

    pub fn generate_leaf_proofs(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
        self.leaf_controller
            .generate_proof(&self.leaf_prover, app_proofs)
    }

    pub fn generate_root_verifier_input(
        &self,
        app_proofs: ContinuationVmProof<SC>,
    ) -> RootVmVerifierInput<SC> {
        let leaf_proofs = self.generate_leaf_proofs(&app_proofs);
        let public_values = app_proofs.user_public_values.public_values;
        let e2e_stark_proof = self.aggregate_leaf_proofs(leaf_proofs, public_values);
        self.wrap_e2e_stark_proof(e2e_stark_proof)
    }

    pub fn aggregate_leaf_proofs(
        &self,
        leaf_proofs: Vec<Proof<SC>>,
        public_values: Vec<F>,
    ) -> VmStarkProof<SC> {
        let mut internal_node_idx = -1;
        let mut internal_node_height = 0;
        let mut proofs = leaf_proofs;
        // We will always generate at least one internal proof, even if there is only one leaf
        // proof, in order to shrink the proof size
        while proofs.len() > 1 || internal_node_height == 0 {
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
        VmStarkProof {
            proof: proofs.pop().unwrap(),
            user_public_values: public_values,
        }
    }

    /// Wrap the e2e stark proof until its heights meet the requirements of the root verifier.
    pub fn wrap_e2e_stark_proof(
        &self,
        e2e_stark_proof: VmStarkProof<SC>,
    ) -> RootVmVerifierInput<SC> {
        let internal_commit = self
            .internal_prover
            .committed_exe
            .get_program_commit()
            .into();
        wrap_e2e_stark_proof(
            &self.internal_prover,
            &self.root_prover,
            internal_commit,
            self.max_internal_wrapper_layers,
            e2e_stark_proof,
        )
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

impl LeafProvingController {
    pub fn with_num_children(mut self, num_children_leaf: usize) -> Self {
        self.num_children = num_children_leaf;
        self
    }

    pub fn generate_proof<E: StarkFriEngine<SC>>(
        &self,
        prover: &VmLocalProver<SC, NativeConfig, E>,
        app_proofs: &ContinuationVmProof<SC>,
    ) -> Vec<Proof<SC>> {
        info_span!("agg_layer", group = "leaf").in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            {
                metrics::counter!("fri.log_blowup").absolute(prover.fri_params().log_blowup as u64);
                metrics::counter!("num_children").absolute(self.num_children as u64);
            }
            let leaf_inputs =
                LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children);
            tracing::info!("num_leaf_proofs={}", leaf_inputs.len());
            leaf_inputs
                .into_iter()
                .enumerate()
                .map(|(leaf_node_idx, input)| {
                    info_span!("single_leaf_agg", idx = leaf_node_idx)
                        .in_scope(|| SingleSegmentVmProver::prove(prover, input.write_to_stream()))
                })
                .collect::<Vec<_>>()
        })
    }
}

/// Wrap the e2e stark proof until its heights meet the requirements of the root verifier.
pub fn wrap_e2e_stark_proof<E: StarkFriEngine<SC>>(
    internal_prover: &VmLocalProver<SC, NativeConfig, E>,
    root_prover: &RootVerifierLocalProver,
    internal_commit: [F; DIGEST_SIZE],
    max_internal_wrapper_layers: usize,
    e2e_stark_proof: VmStarkProof<SC>,
) -> RootVmVerifierInput<SC> {
    let VmStarkProof {
        mut proof,
        user_public_values,
    } = e2e_stark_proof;
    let mut wrapper_layers = 0;
    loop {
        let actual_air_heights = root_prover.execute_for_air_heights(RootVmVerifierInput {
            proofs: vec![proof.clone()],
            public_values: user_public_values.clone(),
        });
        // Root verifier can handle the internal proof. We can stop here.
        if heights_le(
            &actual_air_heights,
            &root_prover.root_verifier_pk.air_heights,
        ) {
            break;
        }
        if wrapper_layers >= max_internal_wrapper_layers {
            panic!("The heights of the root verifier still exceed the required heights after {} wrapper layers", max_internal_wrapper_layers);
        }
        wrapper_layers += 1;
        let input = InternalVmVerifierInput {
            self_program_commit: internal_commit,
            proofs: vec![proof.clone()],
        };
        proof = info_span!(
            "wrapper_layer",
            group = format!("internal_wrapper.{wrapper_layers}")
        )
        .in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            {
                metrics::counter!("fri.log_blowup")
                    .absolute(internal_prover.fri_params().log_blowup as u64);
            }
            SingleSegmentVmProver::prove(internal_prover, input.write())
        });
    }
    RootVmVerifierInput {
        proofs: vec![proof],
        public_values: user_public_values,
    }
}

fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}
