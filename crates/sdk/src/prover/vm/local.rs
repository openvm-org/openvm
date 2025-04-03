#[cfg(feature = "pipeline")]
use std::sync::mpsc;
use std::{marker::PhantomData, mem, sync::Arc};

use async_trait::async_trait;
#[cfg(feature = "pipeline")]
use openvm_circuit::arch::{ExecutionSegment, VmExecutorNextSegmentState};
#[cfg(feature = "pipeline")]
use openvm_circuit::system::memory::paged_vec::AddressMap;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, GenerationError, SingleSegmentVmExecutor, Streams,
        VirtualMachine, VmComplexTraceHeights, VmConfig,
    },
    system::{memory::tree::public_values::UserPublicValuesProof, program::trace::VmCommittedExe},
};
#[cfg(feature = "pipeline")]
use openvm_stark_backend::prover::types::ProofInput;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    proof::Proof,
    Chip,
};
use openvm_stark_sdk::{config::FriParameters, engine::StarkFriEngine};
use tracing::info_span;

use crate::prover::vm::{
    types::VmProvingKey, AsyncContinuationVmProver, AsyncSingleSegmentVmProver,
    ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
};

pub struct VmLocalProver<SC: StarkGenericConfig, VC, E: StarkFriEngine<SC>> {
    pub pk: Arc<VmProvingKey<SC, VC>>,
    pub committed_exe: Arc<VmCommittedExe<SC>>,
    overridden_heights: Option<VmComplexTraceHeights>,
    _marker: PhantomData<E>,
}

impl<SC: StarkGenericConfig, VC, E: StarkFriEngine<SC>> VmLocalProver<SC, VC, E> {
    pub fn new(pk: Arc<VmProvingKey<SC, VC>>, committed_exe: Arc<VmCommittedExe<SC>>) -> Self {
        Self {
            pk,
            committed_exe,
            overridden_heights: None,
            _marker: PhantomData,
        }
    }

    pub fn new_with_overridden_trace_heights(
        pk: Arc<VmProvingKey<SC, VC>>,
        committed_exe: Arc<VmCommittedExe<SC>>,
        overridden_heights: Option<VmComplexTraceHeights>,
    ) -> Self {
        Self {
            pk,
            committed_exe,
            overridden_heights,
            _marker: PhantomData,
        }
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.overridden_heights = Some(overridden_heights);
    }

    pub fn vm_config(&self) -> &VC {
        &self.pk.vm_config
    }
    #[allow(dead_code)]
    pub(crate) fn fri_params(&self) -> &FriParameters {
        &self.pk.fri_params
    }
}

const MAX_SEGMENTATION_RETRIES: usize = 4;

impl<
        SC: StarkGenericConfig + 'static,
        VC: VmConfig<Val<SC>> + 'static + Send,
        E: StarkFriEngine<SC> + 'static + Send,
    > ContinuationVmProver<SC> for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + Send,
    VC::Periphery: Chip<SC> + Send,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> ContinuationVmProof<SC> {
        assert!(self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        let trace_height_constraints = self.pk.vm_pk.trace_height_constraints.clone();
        let mut vm = VirtualMachine::new_with_overridden_trace_heights(
            e,
            self.pk.vm_config.clone(),
            self.overridden_heights.clone(),
        );
        vm.set_trace_height_constraints(trace_height_constraints.clone());
        let VmCommittedExe {
            exe,
            committed_program,
        } = self.committed_exe.as_ref();
        let input = input.into();

        // This loop should typically iterate exactly once. Only in exceptional cases will the
        // segmentation produce an invalid segment and we will have to retry.
        let mut retries = 0;

        #[cfg(not(feature = "pipeline"))]
        let mut final_memory = None;
        #[cfg(not(feature = "pipeline"))]
        let per_segment = loop {
            match vm.executor.execute_and_then(
                exe.clone(),
                input.clone(),
                |seg_idx, mut seg| {
                    final_memory = mem::take(&mut seg.final_memory);
                    let proof_input = info_span!("trace_gen", segment = seg_idx)
                        .in_scope(|| seg.generate_proof_input(Some(committed_program.clone())))?;
                    info_span!("prove_segment", segment = seg_idx)
                        .in_scope(|| Ok(vm.engine.prove(&self.pk.vm_pk, proof_input)))
                },
                GenerationError::Execution,
            ) {
                Ok(per_segment) => break per_segment,
                Err(GenerationError::Execution(err)) => panic!("execution error: {err}"),
                Err(GenerationError::TraceHeightsLimitExceeded) => {
                    if retries >= MAX_SEGMENTATION_RETRIES {
                        panic!(
                            "trace heights limit exceeded after {MAX_SEGMENTATION_RETRIES} retries"
                        );
                    }
                    retries += 1;
                    tracing::info!(
                        "trace heights limit exceeded; retrying execution (attempt {retries})"
                    );
                    let sys_config = vm.executor.config.system_mut();
                    let new_seg_strat = sys_config.segmentation_strategy.stricter_strategy();
                    sys_config.set_segmentation_strategy(new_seg_strat);
                    // continue
                }
            };
        };

        // TODO: add GenerationError
        #[cfg(feature = "pipeline")]
        let (per_segment, final_memory) = {
            tracing::info!("ContinuationVmProver::prove() pipeline feature enabled");
            let span = tracing::Span::current();
            // Set up pipeline channels with capacity 1 to ensure flow control
            let (segment_tx, segment_rx) =
                mpsc::sync_channel::<(usize, ExecutionSegment<Val<SC>, VC>, tracing::Span)>(1);
            let (trace_tx, trace_rx) =
                mpsc::sync_channel::<(usize, ProofInput<SC>, tracing::Span)>(1);
            let (proof_tx, proof_rx) = mpsc::sync_channel::<(usize, Proof<SC>)>(1);
            let (memory_tx, memory_rx) = mpsc::sync_channel(1);

            std::thread::scope(|s| {
                // ===== EXECUTION THREAD =====
                // Executes VM code segments and feeds them to the trace generation pipeline
                let exe_clone = exe.clone();
                let current_span = span.clone();
                let executor_handle = s.spawn(move || {
                    let _guard = current_span.entered();
                    let executor = &vm.executor;
                    let mem_config = executor.config.system().memory_config;
                    let exe_val = exe_clone;

                    let memory = AddressMap::from_iter(
                        mem_config.as_offset,
                        1 << mem_config.as_height,
                        1 << mem_config.pointer_max_bits,
                        exe_val.init_memory.clone(),
                    );
                    let pc = exe_val.pc_start;
                    let mut state = VmExecutorNextSegmentState::new(memory, input, pc);
                    let mut segment_idx = 0;
                    let mut segment_indices = vec![];
                    let mut final_memory = None;

                    // Execute segments until completion
                    loop {
                        let segment_span = info_span!("execute_segment", segment = segment_idx);
                        let _guard = segment_span.enter();

                        // Execute current segment
                        let mut segment_result =
                            match executor.execute_until_segment(exe_val.clone(), state) {
                                Ok(result) => result,
                                Err(e) => panic!("Execution error: {:?}", e),
                            };

                        // Check if this is the final segment
                        let is_last_segment = segment_result.next_state.is_none();
                        if is_last_segment {
                            final_memory = mem::take(&mut segment_result.segment.final_memory);
                        }

                        // Send segment data to trace thread
                        segment_tx
                            .send((
                                segment_idx,
                                segment_result.segment,
                                tracing::Span::current(),
                            ))
                            .expect("Failed to send segment");

                        segment_indices.push(segment_idx);

                        // Exit loop if this was the last segment
                        if is_last_segment {
                            break;
                        }

                        // Prepare for next segment
                        state = segment_result.next_state.unwrap();
                        segment_idx += 1;
                    }
                    drop(segment_tx);

                    // Send final memory state to main thread
                    if let Some(mem) = final_memory {
                        memory_tx.send(mem).expect("Failed to send final memory");
                    } else {
                        panic!("No final memory captured");
                    }

                    segment_indices
                });

                // ===== TRACE GENERATION THREAD =====
                // Generates proof inputs from execution segments
                let current_span = span.clone();
                let trace_handle = s.spawn(move || {
                    let _guard = current_span.entered();
                    let committed_program_clone = committed_program.clone();
                    for (segment_idx, segment, parent_span) in segment_rx.iter() {
                        let span =
                            parent_span.in_scope(|| info_span!("trace_gen", segment = segment_idx));
                        let _guard = span.enter();

                        let proof_input =
                            segment.generate_proof_input(Some(committed_program_clone.clone()));

                        trace_tx
                            .send((
                                segment_idx,
                                proof_input.expect("Failed to generate proof input"),
                                parent_span.clone(),
                            ))
                            .expect("Failed to send trace data");
                    }
                    drop(trace_tx);
                });

                // ===== PROOF GENERATION THREAD =====
                // Generates cryptographic proofs from trace data
                let prove_engine = E::new(self.pk.fri_params);
                let vm_pk_clone = self.pk.vm_pk.clone();
                let current_span = span.clone();
                let prove_handle = s.spawn(move || {
                    let _guard = current_span.entered();
                    for (segment_idx, proof_input, parent_span) in trace_rx.iter() {
                        let span = parent_span
                            .in_scope(|| info_span!("prove_segment", segment = segment_idx));
                        let _guard = span.enter();

                        let proof = prove_engine.prove(&vm_pk_clone, proof_input);

                        proof_tx
                            .send((segment_idx, proof))
                            .expect("Failed to send proof");
                    }
                    drop(proof_tx);
                });

                // ===== COLLECTOR THREAD =====
                // Collects proofs as they are generated
                let current_span = span.clone();
                let collector_handle = s.spawn(move || {
                    let _guard = current_span.entered();
                    let mut proofs = Vec::new();
                    for (_, proof) in proof_rx.iter() {
                        proofs.push(proof); // Collect proofs in order
                    }

                    proofs
                });

                // ===== MAIN THREAD COORDINATION =====
                // Wait for execution & tracegen & prove to complete
                let segment_indices = executor_handle.join().expect("Executor thread panicked");
                let final_memory = memory_rx.recv().expect("Failed to receive final memory");
                trace_handle.join().expect("Trace thread panicked");
                prove_handle.join().expect("Prove thread panicked");

                // Get collected proofs
                let collected_proofs = collector_handle.join().expect("Collector thread panicked");
                if collected_proofs.len() != segment_indices.len() {
                    panic!(
                        "Proof count mismatch: expected {}, got {}",
                        segment_indices.len(),
                        collected_proofs.len()
                    );
                }

                (collected_proofs, Some(final_memory))
            })
        };

        let user_public_values = UserPublicValuesProof::compute(
            self.pk.vm_config.system().memory_config.memory_dimensions(),
            self.pk.vm_config.system().num_public_values,
            &vm_poseidon2_hasher(),
            final_memory.as_ref().unwrap(),
        );
        ContinuationVmProof {
            per_segment,
            user_public_values,
        }
    }
}

impl<
        SC: StarkGenericConfig + 'static,
        VC: VmConfig<Val<SC>> + Send + Sync + 'static,
        E: StarkFriEngine<SC> + Send + 'static,
    > SingleSegmentVmProver<SC> for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + Send,
    VC::Periphery: Chip<SC> + Send,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> Proof<SC> {
        assert!(!self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        // note: use SingleSegmentVmExecutor so there's not a "segment" label in metrics
        let executor = SingleSegmentVmExecutor::new(self.pk.vm_config.clone());
        let proof_input = executor
            .execute_and_generate(self.committed_exe.clone(), input)
            .unwrap();
        let vm = VirtualMachine::new(e, executor.config);
        vm.prove_single(&self.pk.vm_pk, proof_input)
    }
}

#[async_trait]
impl<
        SC: StarkGenericConfig + 'static,
        VC: VmConfig<Val<SC>> + Send + 'static,
        E: StarkFriEngine<SC> + Send + 'static,
    > AsyncContinuationVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + Send,
    VC::Periphery: Chip<SC> + Send,
{
    async fn prove(
        &self,
        input: impl Into<Streams<Val<SC>>> + Send + Sync,
    ) -> ContinuationVmProof<SC> {
        ContinuationVmProver::prove(self, input)
    }
}

#[async_trait]
impl<
        SC: StarkGenericConfig + 'static,
        VC: VmConfig<Val<SC>> + Send + Sync + 'static,
        E: StarkFriEngine<SC> + Send + 'static,
    > AsyncSingleSegmentVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + Send,
    VC::Periphery: Chip<SC> + Send,
{
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
