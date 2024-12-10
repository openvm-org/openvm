use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::prover::types::Proof, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
};
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::SingleSegmentVmExecutor;
use axvm_circuit::{
    arch::Streams,
    prover::{local::VmLocalProver, ContinuationVmProof, SingleSegmentVmProver},
};
use axvm_native_circuit::NativeConfig;
use axvm_native_recursion::hints::Hintable;
use tracing::info_span;

use crate::{
    keygen::AggProvingKey,
    prover::RootVerifierLocalProver,
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
    NonRootCommittedExe, OuterSC, F, SC,
};

const DEFAULT_NUM_CHILDREN_LEAF: usize = 2;
const DEFAULT_NUM_CHILDREN_INTERNAL: usize = 2;
const DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS: usize = 4;

pub struct AggStarkProver {
    leaf_prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    internal_prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    root_prover: RootVerifierLocalProver,

    pub num_children_leaf: usize,
    pub num_children_internal: usize,
    pub max_internal_wrapper_layers: usize,
}
impl AggStarkProver {
    pub fn new(agg_pk: AggProvingKey, leaf_committed_exe: Arc<NonRootCommittedExe>) -> Self {
        let leaf_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            agg_pk.leaf_vm_pk.clone(),
            leaf_committed_exe.clone(),
        );
        let internal_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            agg_pk.internal_vm_pk.clone(),
            agg_pk.internal_committed_exe.clone(),
        );
        let root_prover = RootVerifierLocalProver::new(agg_pk.root_verifier_pk.clone());
        Self {
            leaf_prover,
            internal_prover,
            root_prover,
            num_children_leaf: DEFAULT_NUM_CHILDREN_LEAF,
            num_children_internal: DEFAULT_NUM_CHILDREN_INTERNAL,
            max_internal_wrapper_layers: DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS,
        }
    }
    pub fn with_num_children_leaf(mut self, num_children_leaf: usize) -> Self {
        self.num_children_leaf = num_children_leaf;
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
    pub fn generate_agg_proof(&self, app_proofs: ContinuationVmProof<SC>) -> Proof<OuterSC> {
        let leaf_proofs = info_span!("leaf verifier", group = "leaf_verifier").in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.leaf_prover.pk.fri_params.log_blowup as u64);
            self.generate_leaf_proof_impl(&app_proofs)
        });
        let public_values = app_proofs.user_public_values.public_values;
        let internal_proof = self.generate_internal_proof_impl(leaf_proofs, &public_values);
        info_span!("root verifier", group = "root_verifier").in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup").absolute(
                self.root_prover
                    .root_verifier_pk
                    .vm_pk
                    .fri_params
                    .log_blowup as u64,
            );
            self.generate_root_proof_impl(RootVmVerifierInput {
                proofs: vec![internal_proof],
                public_values,
            })
        })
    }

    fn generate_leaf_proof_impl(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
        let leaf_inputs =
            LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children_leaf);
        leaf_inputs
            .into_iter()
            .enumerate()
            .map(|(leaf_node_idx, input)| {
                info_span!("leaf verifier proof", index = leaf_node_idx)
                    .in_scope(|| single_segment_prove(&self.leaf_prover, input.write_to_stream()))
            })
            .collect::<Vec<_>>()
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
            let group = format!("internal_verifier_height_{}", internal_node_height);
            proofs = info_span!("internal verifier", group = group).in_scope(|| {
                #[cfg(feature = "bench-metrics")]
                metrics::counter!("fri.log_blowup")
                    .absolute(self.internal_prover.pk.fri_params.log_blowup as u64);
                internal_inputs
                    .into_iter()
                    .map(|input| {
                        internal_node_idx += 1;
                        info_span!(
                            "Internal verifier proof",
                            index = internal_node_idx,
                            height = internal_node_height
                        )
                        .in_scope(|| single_segment_prove(&self.internal_prover, input.write()))
                    })
                    .collect()
            });
            internal_node_height += 1;
        }
        proofs.pop().unwrap()
    }

    fn generate_root_proof_impl(&self, root_input: RootVmVerifierInput<SC>) -> Proof<OuterSC> {
        let input = root_input.write();
        let root_prover = &self.root_prover;
        #[cfg(feature = "bench-metrics")]
        {
            let mut vm_config = root_prover.root_verifier_pk.vm_pk.vm_config.clone();
            vm_config.system.collect_metrics = true;
            let vm = SingleSegmentVmExecutor::new(vm_config);
            let exe = root_prover.root_verifier_pk.root_committed_exe.exe.clone();
            vm.execute(exe, input.clone()).unwrap();
        }
        SingleSegmentVmProver::prove(root_prover, input)
    }
}

fn single_segment_prove<E: StarkFriEngine<SC>>(
    prover: &VmLocalProver<SC, NativeConfig, E>,
    input: impl Into<Streams<F>> + Clone,
) -> Proof<SC> {
    #[cfg(feature = "bench-metrics")]
    {
        let mut vm_config = prover.pk.vm_config.clone();
        vm_config.system.collect_metrics = true;
        let vm = SingleSegmentVmExecutor::new(vm_config);
        vm.execute(prover.committed_exe.exe.clone(), input.clone())
            .unwrap();
    }
    SingleSegmentVmProver::prove(prover, input)
}
fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}
