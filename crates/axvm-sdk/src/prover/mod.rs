use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{prover::types::Proof, Chip},
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
};
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::{SingleSegmentVmExecutor, VmExecutor};
use axvm_circuit::{
    arch::{Streams, VmConfig},
    prover::{
        local::VmLocalProver, ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
    },
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_circuit::NativeConfig;
use axvm_native_recursion::hints::Hintable;
use metrics::counter;
use tracing::info_span;

use crate::{
    config::AggConfig,
    io::StdIn,
    keygen::{AggProvingKey, AppProvingKey},
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
    OuterSC, F, SC,
};

mod exe;
pub use exe::*;
mod root;
pub use root::*;

const DEFAULT_NUM_CHILDREN_LEAF: usize = 2;
const DEFAULT_NUM_CHILDREN_INTERNAL: usize = 2;
const DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS: usize = 4;

pub struct StarkProver<VC> {
    pub app_pk: AppProvingKey<VC>,
    pub app_committed_exe: Arc<AxVmCommittedExe<SC>>,
    app_prover: VmLocalProver<SC, VC, BabyBearPoseidon2Engine>,

    pub agg_pk: Option<AggProvingKey>,
    pub leaf_committed_exe: Option<Arc<AxVmCommittedExe<SC>>>,
    leaf_prover: Option<VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>>,
    internal_prover: Option<VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>>,
    root_prover: Option<RootVerifierLocalProver>,

    pub num_children_leaf: usize,
    pub num_children_internal: usize,
    pub max_internal_wrapper_layers: usize,
}

impl<VC: VmConfig<F>> StarkProver<VC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn new(app_pk: AppProvingKey<VC>, app_committed_exe: Arc<AxVmCommittedExe<SC>>) -> Self {
        let app_prover = VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new(
            app_pk.app_vm_pk.clone(),
            app_committed_exe.clone(),
        );
        Self {
            app_pk,
            agg_pk: None,
            app_committed_exe,
            leaf_committed_exe: None,
            num_children_leaf: DEFAULT_NUM_CHILDREN_LEAF,
            num_children_internal: DEFAULT_NUM_CHILDREN_INTERNAL,
            max_internal_wrapper_layers: DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS,
            app_prover,
            leaf_prover: None,
            internal_prover: None,
            root_prover: None,
        }
    }

    pub fn with_agg_config(self, agg_config: AggConfig) -> Self {
        let leaf_committed_exe = generate_leaf_committed_exe(&agg_config, &self.app_pk);
        let agg_pk = AggProvingKey::keygen(agg_config);
        self.with_agg_pk_and_leaf_committed_exe(agg_pk, leaf_committed_exe)
    }

    pub fn with_agg_pk_and_leaf_committed_exe(
        mut self,
        agg_pk: AggProvingKey,
        leaf_committed_exe: Arc<AxVmCommittedExe<SC>>,
    ) -> Self {
        assert_eq!(self.app_pk.num_public_values(), agg_pk.num_public_values());

        let leaf_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            agg_pk.leaf_vm_pk.clone(),
            leaf_committed_exe.clone(),
        );
        let internal_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
            agg_pk.internal_vm_pk.clone(),
            agg_pk.internal_committed_exe.clone(),
        );
        let root_prover = RootVerifierLocalProver::new(agg_pk.root_verifier_pk.clone());

        self.agg_pk = Some(agg_pk);
        self.leaf_committed_exe = Some(leaf_committed_exe);
        self.leaf_prover = Some(leaf_prover);
        self.internal_prover = Some(internal_prover);
        self.root_prover = Some(root_prover);
        self
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

    pub fn agg_pk(&self) -> &AggProvingKey {
        assert!(self.agg_pk.is_some(), "Aggregation has not been configured");
        self.agg_pk.as_ref().unwrap()
    }

    pub fn leaf_committed_exe(&self) -> &Arc<AxVmCommittedExe<SC>> {
        assert!(
            self.leaf_committed_exe.is_some(),
            "Aggregation has not been configured"
        );
        self.leaf_committed_exe.as_ref().unwrap()
    }

    pub fn generate_e2e_proof(&self, input: StdIn) -> Proof<OuterSC> {
        assert!(self.agg_pk.is_some(), "Aggregation has not been configured");
        let app_proofs = self.generate_app_proof(input);
        let leaf_proofs = self.generate_leaf_proof_impl(&app_proofs);
        let public_values = app_proofs.user_public_values.public_values;
        let internal_proof = self.generate_internal_proof_impl(leaf_proofs, &public_values);
        self.generate_root_proof_impl(RootVmVerifierInput {
            proofs: vec![internal_proof],
            public_values,
        })
    }

    pub fn generate_app_proof(&self, input: StdIn) -> ContinuationVmProof<SC> {
        #[cfg(feature = "bench-metrics")]
        {
            execute_app_exe_for_metrics_collection(
                &self.app_pk,
                &self.app_committed_exe,
                input.clone(),
            );
        }
        ContinuationVmProver::prove(&self.app_prover, input)
    }

    pub fn generate_e2e_proof_with_metric_spans(
        &self,
        input: StdIn,
        program_name: &str,
    ) -> Proof<OuterSC> {
        assert!(self.agg_pk.is_some(), "Aggregation has not been configured");
        let app_proofs = self.generate_app_proof_with_metric_spans(input, program_name);
        let leaf_proofs = info_span!("leaf verifier", group = "leaf_verifier").in_scope(|| {
            counter!("fri.log_blowup")
                .absolute(self.agg_pk().leaf_vm_pk.fri_params.log_blowup as u64);
            self.generate_leaf_proof_impl(&app_proofs)
        });
        let public_values = app_proofs.user_public_values.public_values;
        let internal_proof = self.generate_internal_proof_impl(leaf_proofs, &public_values);
        info_span!("root verifier", group = "root_verifier").in_scope(|| {
            counter!("fri.log_blowup")
                .absolute(self.agg_pk().root_verifier_pk.vm_pk.fri_params.log_blowup as u64);
            self.generate_root_proof_impl(RootVmVerifierInput {
                proofs: vec![internal_proof],
                public_values,
            })
        })
    }

    pub fn generate_app_proof_with_metric_spans(
        &self,
        input: StdIn,
        program_name: &str,
    ) -> ContinuationVmProof<SC> {
        let group_name = program_name.replace(" ", "_").to_lowercase();
        info_span!("App Continuation Program", group = group_name).in_scope(|| {
            counter!("fri.log_blowup").absolute(self.app_pk.app_vm_pk.fri_params.log_blowup as u64);
            self.generate_app_proof(input)
        })
    }

    fn generate_leaf_proof_impl(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
        let leaf_inputs =
            LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children_leaf);
        leaf_inputs
            .into_iter()
            .enumerate()
            .map(|(leaf_node_idx, input)| {
                info_span!("leaf verifier proof", index = leaf_node_idx).in_scope(|| {
                    single_segment_prove(
                        self.leaf_prover.as_ref().unwrap(),
                        input.write_to_stream(),
                    )
                })
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
                let root_prover = self.root_prover.as_ref().unwrap();
                // TODO: record execution time as a part of root verifier execution time.
                let actual_air_heights = root_prover.execute_for_air_heights(RootVmVerifierInput {
                    proofs: vec![proofs[0].clone()],
                    public_values: public_values.to_vec(),
                });
                // Root verifier can handle the internal proof. We can stop here.
                if heights_le(
                    &actual_air_heights,
                    &root_prover.root_verifier_pk.air_heights,
                ) {
                    break;
                }
                if wrapper_layers >= self.max_internal_wrapper_layers {
                    panic!("The heights of the root verifier still exceed the required heights after {} wrapper layers", self.max_internal_wrapper_layers);
                }
                wrapper_layers += 1;
            }
            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                self.agg_pk()
                    .internal_committed_exe
                    .get_program_commit()
                    .into(),
                &proofs,
                self.num_children_internal,
            );
            let group = format!("internal_verifier_height_{}", internal_node_height);
            proofs = info_span!("internal verifier", group = group).in_scope(|| {
                counter!("fri.log_blowup")
                    .absolute(self.agg_pk().internal_vm_pk.fri_params.log_blowup as u64);
                internal_inputs
                    .into_iter()
                    .map(|input| {
                        internal_node_idx += 1;
                        info_span!(
                            "Internal verifier proof",
                            index = internal_node_idx,
                            height = internal_node_height
                        )
                        .in_scope(|| {
                            single_segment_prove(
                                self.internal_prover.as_ref().unwrap(),
                                input.write(),
                            )
                        })
                    })
                    .collect()
            });
            internal_node_height += 1;
        }
        proofs.pop().unwrap()
    }

    fn generate_root_proof_impl(&self, root_input: RootVmVerifierInput<SC>) -> Proof<OuterSC> {
        let input = root_input.write();
        let root_prover = self.root_prover.as_ref().unwrap();
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

#[cfg(feature = "bench-metrics")]
fn execute_app_exe_for_metrics_collection<VC: VmConfig<F>>(
    app_pk: &AppProvingKey<VC>,
    app_committed_exe: &Arc<AxVmCommittedExe<SC>>,
    input: StdIn,
) where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let mut vm_config = app_pk.app_vm_pk.vm_config.clone();
    vm_config.system_mut().collect_metrics = true;
    let vm = VmExecutor::new(vm_config);
    vm.execute_segments(app_committed_exe.exe.clone(), input)
        .unwrap();
}

fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}
