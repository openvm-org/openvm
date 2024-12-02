use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{prover::types::Proof, Chip},
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
};
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::{SingleSegmentVmExecutor, VmExecutor};
use axvm_circuit::{
    arch::VmConfig,
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
    keygen::{AggProvingKey, AppProvingKey},
    prover::RootVerifierLocalProver,
    verifier::{
        internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
        root::types::RootVmVerifierInput,
    },
    OuterSC, F, SC,
};

mod exe;
pub use exe::*;

pub struct E2EStarkProver<VC> {
    pub app_pk: AppProvingKey<VC>,
    pub agg_pk: AggProvingKey,
    pub app_committed_exe: Arc<AxVmCommittedExe<SC>>,
    pub leaf_committed_exe: Arc<AxVmCommittedExe<SC>>,

    pub num_children_leaf: usize,
    pub num_children_internal: usize,

    app_prover: VmLocalProver<SC, VC, BabyBearPoseidon2Engine>,
    leaf_prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    internal_prover: VmLocalProver<SC, NativeConfig, BabyBearPoseidon2Engine>,
    root_prover: RootVerifierLocalProver,
}

impl<VC: VmConfig<F>> E2EStarkProver<VC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn new(
        app_pk: AppProvingKey<VC>,
        agg_pk: AggProvingKey,
        app_committed_exe: Arc<AxVmCommittedExe<SC>>,
        leaf_committed_exe: Arc<AxVmCommittedExe<SC>>,
        num_children_leaf: usize,
        num_children_internal: usize,
    ) -> Self {
        assert_eq!(app_pk.num_public_values(), agg_pk.num_public_values());
        let app_prover = VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new(
            app_pk.app_vm_pk.clone(),
            app_committed_exe.clone(),
        );
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
            app_pk,
            agg_pk,
            app_committed_exe,
            leaf_committed_exe,
            num_children_leaf,
            num_children_internal,
            app_prover,
            leaf_prover,
            internal_prover,
            root_prover,
        }
    }

    pub fn generate_proof(&self, input: Vec<F>) -> Proof<OuterSC> {
        let app_proofs = self.generate_app_proof(input);
        let leaf_proofs = self.generate_leaf_proof(&app_proofs);
        let internal_proof = self.generate_internal_proof(leaf_proofs);
        self.generate_root_proof(app_proofs, internal_proof)
    }

    pub fn generate_proof_with_metric_spans(
        &self,
        input: Vec<F>,
        program_name: &str,
    ) -> Proof<OuterSC> {
        let group_name = program_name.replace(" ", "_").to_lowercase();
        let app_proofs =
            info_span!("App Continuation Program", group = group_name).in_scope(|| {
                counter!("fri.log_blowup")
                    .absolute(self.app_pk.app_vm_pk.fri_params.log_blowup as u64);
                self.generate_app_proof(input)
            });
        let leaf_proofs = info_span!("leaf verifier", group = "leaf_verifier").in_scope(|| {
            counter!("fri.log_blowup")
                .absolute(self.agg_pk.leaf_vm_pk.fri_params.log_blowup as u64);
            self.generate_leaf_proof(&app_proofs)
        });
        let internal_proof = self.generate_internal_proof(leaf_proofs);
        info_span!("root verifier", group = "root_verifier").in_scope(|| {
            counter!("fri.log_blowup")
                .absolute(self.agg_pk.root_verifier_pk.vm_pk.fri_params.log_blowup as u64);
            self.generate_root_proof(app_proofs, internal_proof)
        })
    }

    fn generate_app_proof(&self, input: Vec<F>) -> ContinuationVmProof<SC> {
        #[cfg(feature = "bench-metrics")]
        {
            let mut vm_config = self.app_pk.app_vm_pk.vm_config.clone();
            vm_config.system_mut().collect_metrics = true;
            let vm = VmExecutor::new(vm_config);
            vm.execute_segments(self.app_committed_exe.exe.clone(), vec![input.clone()])
                .unwrap();
        }
        ContinuationVmProver::prove(&self.app_prover, vec![input])
    }

    fn generate_leaf_proof(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
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

    fn generate_internal_proof(&self, leaf_proofs: Vec<Proof<SC>>) -> Proof<SC> {
        let mut internal_node_idx = -1;
        let mut internal_node_height = 0;
        let mut proofs = leaf_proofs;
        while proofs.len() > 1 {
            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                self.agg_pk
                    .internal_committed_exe
                    .get_program_commit()
                    .into(),
                &proofs,
                self.num_children_internal,
            );
            let group = format!("internal_verifier_height_{}", internal_node_height);
            proofs = info_span!("internal verifier", group = group).in_scope(|| {
                counter!("fri.log_blowup")
                    .absolute(self.agg_pk.internal_vm_pk.fri_params.log_blowup as u64);
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

    fn generate_root_proof(
        &self,
        app_proofs: ContinuationVmProof<SC>,
        internal_proof: Proof<SC>,
    ) -> Proof<OuterSC> {
        // TODO: wrap internal verifier if heights exceed
        let root_input = RootVmVerifierInput {
            proofs: vec![internal_proof],
            public_values: app_proofs.user_public_values.public_values,
        };
        let input = root_input.write();
        #[cfg(feature = "bench-metrics")]
        {
            let mut vm_config = self.root_prover.root_verifier_pk.vm_pk.vm_config.clone();
            vm_config.system.collect_metrics = true;
            let vm = SingleSegmentVmExecutor::new(vm_config);
            let exe = self
                .root_prover
                .root_verifier_pk
                .root_committed_exe
                .exe
                .clone();
            vm.execute(exe, input.clone()).unwrap();
        }
        SingleSegmentVmProver::prove(&self.root_prover, input)
    }
}

fn single_segment_prove<E: StarkFriEngine<SC>>(
    prover: &VmLocalProver<SC, NativeConfig, E>,
    input: Vec<Vec<F>>,
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
