use std::sync::{mpsc, Arc};

use openvm_circuit::arch::{ContinuationVmProof, SingleSegmentVmExecutor, Streams, VirtualMachine};
use openvm_continuations::verifier::{
    internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput,
    root::types::RootVmVerifierInput,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::ProofInput,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    openvm_stark_backend::proof::Proof,
};
use tracing::info_span;

use crate::{
    keygen::AggStarkProvingKey,
    prover::{
        vm::{local::VmLocalProver, SingleSegmentVmProver},
        RootVerifierLocalProver,
    },
    NonRootCommittedExe, RootSC, F, SC,
};

pub const DEFAULT_NUM_CHILDREN_LEAF: usize = 1;
const DEFAULT_NUM_CHILDREN_INTERNAL: usize = 2;
const DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS: usize = 4;

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
    ) -> Self {
        let leaf_prover =
            VmLocalProver::<SC, NativeConfig, E>::new(agg_stark_pk.leaf_vm_pk, leaf_committed_exe);
        let leaf_controller = LeafProvingController {
            num_children: DEFAULT_NUM_CHILDREN_LEAF,
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
            num_children_internal: DEFAULT_NUM_CHILDREN_INTERNAL,
            max_internal_wrapper_layers: DEFAULT_MAX_INTERNAL_WRAPPER_LAYERS,
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

    /// Generate a proof to aggregate app proofs.
    pub fn generate_agg_proof(&self, app_proofs: ContinuationVmProof<SC>) -> Proof<RootSC> {
        let root_verifier_input = self.generate_root_verifier_input(app_proofs);
        self.generate_root_proof_impl(root_verifier_input)
    }

    pub fn generate_root_verifier_input(
        &self,
        app_proofs: ContinuationVmProof<SC>,
    ) -> RootVmVerifierInput<SC> {
        let leaf_proofs = self
            .leaf_controller
            .generate_proof(&self.leaf_prover, &app_proofs);
        let public_values = app_proofs.user_public_values.public_values;
        let internal_proof = self.generate_internal_proof_impl(leaf_proofs, &public_values);
        RootVmVerifierInput {
            proofs: vec![internal_proof],
            public_values,
        }
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
            let wall_timer = std::time::Instant::now();
            if proofs.len() == 1 {
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
            let internal_inputs_num = internal_inputs.len();

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

                // Use parallel processing with expanded SingleSegmentVmProver::prove
                std::thread::scope(|s| {
                    // Create channels
                    let (input_tx, input_rx) =
                        mpsc::sync_channel::<(usize, Streams<Val<SC>>, tracing::Span)>(1);
                    let (proof_input_tx, proof_input_rx) =
                        mpsc::sync_channel::<(usize, ProofInput<SC>, tracing::Span)>(1);
                    let (proof_tx, proof_rx) = mpsc::sync_channel::<(usize, Proof<SC>)>(1);

                    // Clone required data
                    let committed_exe = self.internal_prover.committed_exe.clone();
                    let vm_config = self.internal_prover.pk.vm_config.clone();
                    let vm_pk = self.internal_prover.pk.clone();
                    let fri_params = self.internal_prover.pk.fri_params;

                    // ===== Trace Generation Thread =====
                    let executor = SingleSegmentVmExecutor::new(vm_config.clone());
                    let trace_handle = s.spawn(move || {
                        for (idx, input, parent_span) in input_rx.iter() {
                            let span = parent_span.in_scope(|| info_span!("trace_gen", idx = idx));
                            let _guard = span.enter();

                            // Execute and generate trace data
                            let proof_input = executor
                                .execute_and_generate(committed_exe.clone(), input)
                                .unwrap();

                            // Send result to proof thread
                            proof_input_tx
                                .send((idx, proof_input, parent_span.clone()))
                                .expect("Failed to send proof input");
                        }

                        drop(proof_input_tx);
                    });

                    // ===== Proof Generation Thread =====
                    let prove_handle = s.spawn(move || {
                        let prove_engine = BabyBearPoseidon2Engine::new(fri_params);
                        let vm = VirtualMachine::new(prove_engine, vm_config.clone());

                        for (idx, proof_input, parent_span) in proof_input_rx.iter() {
                            let span =
                                parent_span.in_scope(|| info_span!("prove_segment", idx = idx));
                            let _guard = span.enter();

                            // Generate proof
                            let proof = vm.prove_single(&vm_pk.vm_pk, proof_input);

                            // Send proof
                            proof_tx.send((idx, proof)).expect("Failed to send proof");
                        }

                        drop(proof_tx);
                    });

                    // ===== Collector Thread =====
                    let collector_handle = s.spawn(move || {
                        let mut proofs = Vec::with_capacity(internal_inputs_num);
                        let mut pending_proofs = std::collections::BTreeMap::new();
                        let mut next_idx = 0;

                        for (idx, proof) in proof_rx.iter() {
                            pending_proofs.insert(idx, proof);

                            // Collect proofs in order
                            while let Some(proof) = pending_proofs.remove(&next_idx) {
                                proofs.push(proof);
                                next_idx += 1;
                            }
                        }

                        // Ensure all proofs have been collected in order
                        let expected = internal_inputs_num;
                        let actual = proofs.len();
                        assert_eq!(
                            actual, expected,
                            "Expected {} proofs, but got {}",
                            expected, actual
                        );

                        proofs
                    });

                    // ===== Main Thread - Send Inputs =====
                    for (idx, input) in internal_inputs.into_iter().enumerate() {
                        internal_node_idx += 1;
                        let span = info_span!("single_internal_agg", idx = internal_node_idx);
                        let _guard = span.enter();

                        // Send input
                        input_tx
                            .send((idx, input.write().into(), span.clone()))
                            .expect("Failed to send internal node input");
                    }

                    drop(input_tx);

                    // Wait for all threads to complete
                    trace_handle.join().expect("Trace thread panicked");
                    prove_handle.join().expect("Prove thread panicked");

                    collector_handle.join().expect("Collector thread panicked")
                })
            });

            internal_node_height += 1;
            tracing::info!(
                "internal.{internal_node_height} wall time: {:?}",
                wall_timer.elapsed()
            );
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
            let wall_timer = std::time::Instant::now();
            #[cfg(feature = "bench-metrics")]
            {
                metrics::counter!("fri.log_blowup").absolute(prover.fri_params().log_blowup as u64);
                metrics::counter!("num_children").absolute(self.num_children as u64);
            }
            let leaf_inputs =
                LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children);
            tracing::info!("num_leaf_proofs={}", leaf_inputs.len());

            let proofs = std::thread::scope(|s| {
                let (input_tx, input_rx) =
                    mpsc::sync_channel::<(Streams<Val<SC>>, tracing::Span)>(1);
                let (proof_input_tx, proof_input_rx) =
                    mpsc::sync_channel::<(ProofInput<SC>, tracing::Span)>(1);
                let (proof_tx, proof_rx) = mpsc::sync_channel::<Proof<SC>>(1);

                let committed_exe = prover.committed_exe.clone();
                let vm_config = prover.pk.vm_config.clone();
                let vm_pk = prover.pk.clone();
                let fri_params = prover.pk.fri_params;

                let executor = SingleSegmentVmExecutor::new(vm_config.clone());
                let trace_handle = s.spawn(move || {
                    for (input, parent_span) in input_rx.iter() {
                        let span = parent_span.in_scope(|| info_span!("trace_gen"));
                        let _guard = span.enter();

                        let proof_input = executor
                            .execute_and_generate(committed_exe.clone(), input)
                            .unwrap();

                        proof_input_tx
                            .send((proof_input, parent_span.clone()))
                            .expect("Failed to send proof input");
                    }
                    drop(proof_input_tx);
                });

                let prove_handle = s.spawn(move || {
                    let prove_engine = BabyBearPoseidon2Engine::new(fri_params);
                    let vm = VirtualMachine::new(prove_engine, vm_config.clone());

                    for (proof_input, parent_span) in proof_input_rx.iter() {
                        let span = parent_span.in_scope(|| info_span!("prove_segment"));
                        let _guard = span.enter();

                        let proof = vm.prove_single(&vm_pk.vm_pk, proof_input);

                        proof_tx.send(proof).expect("Failed to send proof");
                    }
                    drop(proof_tx);
                });

                let collector_handle = s.spawn(move || {
                    let mut proofs = Vec::new();
                    for proof in proof_rx.iter() {
                        proofs.push(proof); // Collect proofs in order
                    }

                    proofs
                });

                for (leaf_node_idx, input) in leaf_inputs.into_iter().enumerate() {
                    let span = info_span!("single_leaf_agg", idx = leaf_node_idx);
                    let _guard = span.enter();

                    input_tx
                        .send((input.write_to_stream().into(), span.clone()))
                        .expect("Failed to send leaf input");
                }

                drop(input_tx);

                trace_handle.join().expect("Trace thread panicked");
                prove_handle.join().expect("Prove thread panicked");

                collector_handle.join().expect("Collector thread panicked")
            });

            tracing::info!("leaf wall time: {:?}", wall_timer.elapsed());
            proofs
        })
    }
}

fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}
