use std::{marker::PhantomData, mem, sync::Arc};

use async_trait::async_trait;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, GenerationError, SingleSegmentVmExecutor, Streams,
        VirtualMachine, VmComplexTraceHeights, VmConfig,
    },
    system::{memory::tree::public_values::UserPublicValuesProof, program::trace::VmCommittedExe},
};
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

impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>> ContinuationVmProver<SC>
    for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
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
        let mut final_memory = None;
        let VmCommittedExe {
            exe,
            committed_program,
        } = self.committed_exe.as_ref();
        let input = input.into();

        // This loop should typically iterate exactly once. Only in exceptional cases will the
        // segmentation produce an invalid segment and we will have to retry.
        let mut retries = 0;
        let per_segment = loop {
            match vm.executor.execute_and_then(
                exe.clone(),
                input.clone(),
                |seg_idx, seg| {
                    final_memory = Some(
                        seg.chip_complex
                            .memory_controller()
                            .memory
                            .data
                            .memory
                            .clone(),
                    );
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

#[async_trait]
impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncContinuationVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    async fn prove(
        &self,
        input: impl Into<Streams<Val<SC>>> + Send + Sync,
    ) -> ContinuationVmProof<SC> {
        ContinuationVmProver::prove(self, input)
    }
}

impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>> SingleSegmentVmProver<SC>
    for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> Proof<SC> {
        assert!(!self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        // note: use SingleSegmentVmExecutor so there's not a "segment" label in metrics
        let executor = {
            let mut executor = SingleSegmentVmExecutor::new(self.pk.vm_config.clone());
            executor.set_trace_height_constraints(self.pk.vm_pk.trace_height_constraints.clone());
            executor
        };
        let proof_input = executor
            .execute_and_generate(self.committed_exe.clone(), input)
            .unwrap();
        let vm = VirtualMachine::new(e, executor.config);
        vm.prove_single(&self.pk.vm_pk, proof_input)
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncSingleSegmentVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
