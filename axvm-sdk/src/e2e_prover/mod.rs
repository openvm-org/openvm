use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{config::Val, prover::types::Proof},
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
};
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::{SingleSegmentVmExecutor, VmExecutor};
use axvm_circuit::{
    prover::{
        local::VmLocalProver, ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
    },
    system::program::trace::AxVmCommittedExe,
};
use axvm_recursion::hints::Hintable;
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

const NUM_CHILDREN_LEAF: usize = 2;
const NUM_CHILDREN_INTERNAL: usize = 2;

mod exe;
pub use exe::*;

pub struct E2EStarkProver {
    pub app_pk: AppProvingKey,
    pub agg_pk: AggProvingKey,
    pub app_committed_exe: Arc<AxVmCommittedExe<SC>>,
    pub leaf_committed_exe: Arc<AxVmCommittedExe<SC>>,
}

impl E2EStarkProver {
    pub fn new(
        app_pk: AppProvingKey,
        agg_pk: AggProvingKey,
        app_committed_exe: Arc<AxVmCommittedExe<SC>>,
        leaf_committed_exe: Arc<AxVmCommittedExe<SC>>,
    ) -> Self {
        assert_eq!(app_pk.num_public_values(), agg_pk.num_public_values());
        Self {
            app_pk,
            agg_pk,
            app_committed_exe,
            leaf_committed_exe,
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
            vm_config.collect_metrics = true;
            let vm = VmExecutor::new(vm_config);
            vm.execute_segments(self.app_committed_exe.exe.clone(), vec![input.clone()])
                .unwrap();
        }
        let app_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
            self.app_pk.app_vm_pk.clone(),
            self.app_committed_exe.clone(),
        );
        ContinuationVmProver::prove(&app_prover, vec![input])
    }

    fn generate_leaf_proof(&self, app_proofs: &ContinuationVmProof<SC>) -> Vec<Proof<SC>> {
        let leaf_inputs =
            LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, NUM_CHILDREN_LEAF);
        let leaf_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
            self.agg_pk.leaf_vm_pk.clone(),
            self.leaf_committed_exe.clone(),
        );
        leaf_inputs
            .into_iter()
            .enumerate()
            .map(|(leaf_node_idx, input)| {
                info_span!("leaf verifier proof", index = leaf_node_idx)
                    .in_scope(|| Self::single_segment_prove(&leaf_prover, input.write_to_stream()))
            })
            .collect::<Vec<_>>()
    }

    fn generate_internal_proof(&self, leaf_proofs: Vec<Proof<SC>>) -> Proof<SC> {
        let internal_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
            self.agg_pk.internal_vm_pk.clone(),
            self.agg_pk.internal_committed_exe.clone(),
        );
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
                NUM_CHILDREN_INTERNAL,
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
                        .in_scope(|| Self::single_segment_prove(&internal_prover, input.write()))
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
        let root_prover = RootVerifierLocalProver::new(self.agg_pk.root_verifier_pk.clone());
        let root_input = RootVmVerifierInput {
            proofs: vec![internal_proof],
            public_values: app_proofs.user_public_values.public_values,
        };
        let input = root_input.write();
        #[cfg(feature = "bench-metrics")]
        {
            let mut vm_config = root_prover.root_verifier_pk.vm_pk.vm_config.clone();
            vm_config.collect_metrics = true;
            let vm = SingleSegmentVmExecutor::new(vm_config);
            let exe = root_prover.root_verifier_pk.root_committed_exe.exe.clone();
            vm.execute(exe, input.clone()).unwrap();
        }
        SingleSegmentVmProver::prove(&root_prover, input)
    }

    fn single_segment_prove<E: StarkFriEngine<SC>>(
        prover: &VmLocalProver<SC, E>,
        input: Vec<Vec<Val<SC>>>,
    ) -> Proof<SC> {
        #[cfg(feature = "bench-metrics")]
        {
            let mut vm_config = prover.pk.vm_config.clone();
            vm_config.collect_metrics = true;
            let vm = SingleSegmentVmExecutor::new(vm_config);
            vm.execute(prover.committed_exe.exe.clone(), input.clone())
                .unwrap();
        }
        SingleSegmentVmProver::prove(prover, input)
    }
}
