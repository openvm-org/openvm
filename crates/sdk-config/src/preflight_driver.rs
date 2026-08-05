//! Record-free CUDA continuation proving.
//!
//! Metered execution fixes segment boundaries. Preflight then converts each
//! segment's mutable execution into immutable program and memory history.
//! Postflight derives GPU replay indexes from that history before trace
//! generation. With `rvr`, only the preflight-history producer changes: the
//! compiled executor emits checkpoints which are expanded on the GPU.

#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::MeteredCtx,
    rvr::{
        PreflightEndpoint, PreflightExecution, PreflightInstance, PreflightTranscript,
        RvrMeteredInstance,
    },
};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{PreflightInterpretedInstance, PreflightOutput, VmFieldExecutionConfig};
use openvm_circuit::{
    arch::{
        cuda::postflight::{GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript},
        execution_mode::Segment,
        hasher::poseidon2::vm_poseidon2_hasher,
        ContinuationVmProof, ExitCode, GenerationError, Streams, VirtualMachine,
        VirtualMachineError, VmInstance, VmState,
    },
    system::memory::{merkle::public_values::UserPublicValuesProof, online::GuestMemory},
};
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
#[cfg(feature = "rvr")]
use openvm_riscv_circuit::preflight::PreflightReplayProgram;
#[cfg(not(feature = "rvr"))]
use openvm_stark_backend::Val;
use openvm_stark_backend::{proof::Proof, StarkEngine};
use tracing::info_span;

#[cfg(not(feature = "rvr"))]
use crate::SdkVmConfig;
use crate::{SdkVmGpuBuilder, SC};

#[cfg(not(feature = "rvr"))]
type InterpretedPreflight =
    PreflightInterpretedInstance<<SdkVmConfig as VmFieldExecutionConfig<Val<SC>>>::Executor>;

/// Fixed-program GPU prover for standalone, independently scheduled segments.
///
/// Unlike continuation proving, each call starts a fresh preflight transcript,
/// so callers may prove segments out of order from arbitrary segment-start
/// states.
pub struct SegmentProver {
    // Drop generated preflight libraries before the VM executor they reference.
    prepared: PreparedSegment,
    instance: VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
}

struct PreparedSegment {
    #[cfg(feature = "rvr")]
    preflight: PreflightInstance<'static>,
    #[cfg(not(feature = "rvr"))]
    preflight: InterpretedPreflight,
    #[cfg(feature = "rvr")]
    program: PreflightReplayProgram,
    #[cfg(not(feature = "rvr"))]
    program: GpuPostflightProgram,
}

impl SegmentProver {
    pub fn new(
        instance: VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let _prepare = info_span!("prepare_preflight", group = "app_proof").entered();
        let prepared = PreparedSegment::new(&instance)?;
        Ok(Self { prepared, instance })
    }

    /// Proves one segment from an arbitrary segment-start state.
    ///
    /// Final memory is returned only when the segment terminates successfully.
    pub fn prove(
        &mut self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<SC>, Option<GuestMemory>), VirtualMachineError> {
        self.prepared.prove(&mut self.instance, state, segment)
    }

    /// Returns the fixed-program VM used by this prover.
    pub fn vm(&self) -> &VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder> {
        &self.instance.vm
    }
}

impl PreparedSegment {
    fn new(
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let exe = instance.exe();

        #[cfg(feature = "rvr")]
        let preflight = instance
            .vm
            .executor()
            .preflight_instance(exe)
            .map_err(VirtualMachineError::from)?
            .into_owned();
        #[cfg(not(feature = "rvr"))]
        let preflight = instance.vm.preflight_interpreter(exe)?;

        let program = info_span!("upload_preflight_program")
            .in_scope(|| SdkVmGpuBuilder::upload_preflight_program(&instance.vm, &exe.program))
            .map_err(generation_error)?;

        Ok(Self { preflight, program })
    }

    fn tracegen_program(&self) -> &GpuPostflightProgram {
        #[cfg(feature = "rvr")]
        {
            self.program.program()
        }
        #[cfg(not(feature = "rvr"))]
        {
            &self.program
        }
    }

    #[cfg(feature = "rvr")]
    fn execute_segment(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
        reuse: Option<PreflightTranscript>,
    ) -> Result<PreflightExecution, VirtualMachineError> {
        match reuse {
            Some(transcript) => self
                .preflight
                .execute_segment_reusing(state, segment, transcript),
            None => self.preflight.execute_segment(state, segment),
        }
        .map_err(VirtualMachineError::from)
    }

    #[cfg(not(feature = "rvr"))]
    fn execute_segment(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightOutput, VirtualMachineError> {
        self.preflight
            .execute_segment(state, segment)
            .map_err(VirtualMachineError::from)
    }

    #[cfg(feature = "rvr")]
    fn postflight(
        &self,
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        execution: &PreflightExecution,
        segment: &Segment,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), VirtualMachineError> {
        let num_insns = u32::try_from(segment.num_insns)
            .map_err(|_| generation_error("metered instruction count exceeds u32"))?;
        SdkVmGpuBuilder::postflight(vm, &self.program, execution, num_insns)
            .map_err(generation_error)
    }

    #[cfg(not(feature = "rvr"))]
    fn postflight(
        &self,
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        output: &mut PreflightOutput,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), VirtualMachineError> {
        vm.postflight_history(&self.program, output)
            .map_err(generation_error)
    }

    fn prove(
        &self,
        instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<SC>, Option<GuestMemory>), VirtualMachineError> {
        let _prove_span = info_span!("total_proof").entered();
        #[cfg(feature = "perf-metrics")]
        let exe = instance.exe().clone();

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);

        #[cfg(feature = "rvr")]
        let (state, exit_code, gpu_transcript, replay_plan) = {
            let execution = self.execute_segment(state, segment, None)?;
            #[cfg(feature = "perf-metrics")]
            let mut execution = execution;
            let exit_code = matches!(&execution.endpoint, PreflightEndpoint::Terminated)
                .then_some(ExitCode::Success as u32);
            let (gpu_transcript, replay_plan) =
                self.postflight(&instance.vm, &execution, segment)?;
            #[cfg(feature = "perf-metrics")]
            instance
                .vm
                .emit_gpu_guest_instruction_metrics(
                    &exe.program,
                    &gpu_transcript,
                    &mut execution.state.metrics,
                )
                .map_err(VirtualMachineError::from)?;
            (execution.state, exit_code, gpu_transcript, replay_plan)
        };

        #[cfg(not(feature = "rvr"))]
        let (state, exit_code, gpu_transcript, replay_plan) = {
            let mut output = self.execute_segment(state, segment)?;
            let (gpu_transcript, replay_plan) = self.postflight(&instance.vm, &mut output)?;
            #[cfg(feature = "perf-metrics")]
            instance
                .vm
                .emit_gpu_guest_instruction_metrics(
                    &exe.program,
                    &gpu_transcript,
                    &mut output.state.metrics,
                )
                .map_err(VirtualMachineError::from)?;
            (output.state, output.exit_code, gpu_transcript, replay_plan)
        };

        let ctx = SdkVmGpuBuilder::generate_preflight_proving_ctx(
            &mut instance.vm,
            self.tracegen_program(),
            &gpu_transcript,
            &replay_plan,
        )?;
        drop(replay_plan);
        drop(gpu_transcript);

        let proof = instance
            .vm
            .engine
            .prove(instance.vm.pk(), ctx)
            .map_err(|error| GenerationError::Proving(error.to_string()))?;
        let final_memory = (exit_code == Some(ExitCode::Success as u32)).then_some(state.memory);
        Ok((proof, final_memory))
    }
}

/// Prepared whole-program continuation prover for the GPU SDK.
pub struct PreparedContinuation {
    #[cfg(feature = "rvr")]
    metered: RvrMeteredInstance<'static>,
    #[cfg(feature = "rvr")]
    metered_ctx: MeteredCtx,
    segment: PreparedSegment,
}

impl PreparedContinuation {
    pub(crate) fn new(
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let _prepare = info_span!("prepare_preflight", group = "app_proof").entered();
        #[cfg(feature = "rvr")]
        let exe = instance.exe();
        let segment = PreparedSegment::new(instance)?;

        #[cfg(feature = "rvr")]
        let (metered, metered_ctx) = {
            let metered_ctx = instance.vm.build_metered_ctx(exe);
            let executor_idx_to_air_idx = instance.vm.executor_idx_to_air_idx();
            let metered = instance
                .vm
                .executor()
                .metered_instance_with_debug_map(
                    exe,
                    &executor_idx_to_air_idx,
                    metered_ctx.trace_heights.len(),
                    None,
                )
                .map_err(VirtualMachineError::from)?
                .into_owned();
            (metered, metered_ctx)
        };

        Ok(Self {
            #[cfg(feature = "rvr")]
            metered,
            #[cfg(feature = "rvr")]
            metered_ctx,
            segment,
        })
    }

    fn tracegen_program(&self) -> &GpuPostflightProgram {
        self.segment.tracegen_program()
    }

    pub(crate) fn prove(
        &mut self,
        instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        input: Streams,
    ) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
        instance.reset_state(input.clone());
        let exe = instance.exe().clone();
        let result = prove_inner(instance, input, self);
        if result.is_err() && instance.state().is_none() {
            // Segment execution consumes the state. Restore an allocated state so
            // this fixed-program prover remains reusable after an error.
            *instance.state_mut() =
                Some(instance.vm.create_initial_state(&exe, Streams::default()));
        }
        result
    }
}

fn prove_inner(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    prepared: &PreparedContinuation,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    #[cfg(feature = "perf-metrics")]
    let exe = instance.exe().clone();
    #[cfg(feature = "rvr")]
    let segments = prepared
        .metered
        .execute_metered(input, prepared.metered_ctx.clone())?
        .0;
    #[cfg(not(feature = "rvr"))]
    let segments = {
        let metered_ctx = instance.vm.build_metered_ctx(instance.exe());
        let metered = instance.vm.metered_instance(instance.exe())?;
        metered.execute_metered(input, metered_ctx)?.0
    };
    let num_segments = segments.len();
    let mut proofs = Vec::with_capacity(num_segments);
    let mut state = instance
        .state_mut()
        .take()
        .ok_or_else(|| generation_error("VM instance has no execution state"))?;
    #[cfg(feature = "rvr")]
    let mut reuse = None;

    for (segment_idx, segment) in segments.into_iter().enumerate() {
        let _segment_span = info_span!("prove_segment", segment = segment_idx).entered();
        let _prove_span = info_span!("total_proof").entered();

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);

        #[cfg(feature = "rvr")]
        let (next_state, gpu_transcript, replay_plan) = {
            let execution = prepared
                .segment
                .execute_segment(state, &segment, reuse.take())?;
            #[cfg(feature = "perf-metrics")]
            let mut execution = execution;
            validate_endpoint(
                matches!(&execution.endpoint, PreflightEndpoint::Terminated),
                segment_idx + 1 == num_segments,
            )?;
            let (gpu_transcript, replay_plan) =
                prepared
                    .segment
                    .postflight(&instance.vm, &execution, &segment)?;
            #[cfg(feature = "perf-metrics")]
            instance
                .vm
                .emit_gpu_guest_instruction_metrics(
                    &exe.program,
                    &gpu_transcript,
                    &mut execution.state.metrics,
                )
                .map_err(VirtualMachineError::from)?;
            let PreflightExecution {
                state: next_state,
                mut transcript,
                ..
            } = execution;
            transcript.checkpoints.clear();
            transcript.replay_values.clear();
            reuse = Some(transcript);
            (next_state, gpu_transcript, replay_plan)
        };

        #[cfg(not(feature = "rvr"))]
        let (next_state, gpu_transcript, replay_plan) = {
            let mut output = prepared.segment.execute_segment(state, &segment)?;
            validate_endpoint(output.exit_code.is_some(), segment_idx + 1 == num_segments)?;
            let (gpu_transcript, replay_plan) =
                prepared.segment.postflight(&instance.vm, &mut output)?;
            #[cfg(feature = "perf-metrics")]
            instance
                .vm
                .emit_gpu_guest_instruction_metrics(
                    &exe.program,
                    &gpu_transcript,
                    &mut output.state.metrics,
                )
                .map_err(VirtualMachineError::from)?;
            (output.state, gpu_transcript, replay_plan)
        };

        state = next_state;
        let ctx = SdkVmGpuBuilder::generate_preflight_proving_ctx(
            &mut instance.vm,
            prepared.tracegen_program(),
            &gpu_transcript,
            &replay_plan,
        )?;

        // Tracegen has synchronized before returning. Release segment-sized
        // replay data before the proving phase reaches its memory peak.
        drop(replay_plan);
        drop(gpu_transcript);

        let proof = instance
            .vm
            .engine
            .prove(instance.vm.pk(), ctx)
            .map_err(|error| GenerationError::Proving(error.to_string()))?;
        proofs.push(proof);
    }

    let final_memory_top_tree = instance
        .vm
        .memory_top_tree()
        .ok_or_else(|| generation_error("final memory top tree was not generated"))?;
    let user_public_values = UserPublicValuesProof::compute(
        instance.vm.config().as_ref(),
        &vm_poseidon2_hasher(),
        &state.memory.memory,
        final_memory_top_tree,
    );
    *instance.state_mut() = Some(state);

    Ok(ContinuationVmProof {
        per_segment: proofs,
        user_public_values,
    })
}

fn validate_endpoint(terminated: bool, is_final_segment: bool) -> Result<(), VirtualMachineError> {
    if terminated == is_final_segment {
        Ok(())
    } else {
        Err(generation_error(if is_final_segment {
            "final metered segment suspended instead of terminating"
        } else {
            "non-final metered segment terminated"
        }))
    }
}

fn generation_error(error: impl ToString) -> VirtualMachineError {
    VirtualMachineError::Generation(GenerationError::ExtensionTracegen(error.to_string()))
}
