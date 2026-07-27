//! Record-free CUDA continuation proving.
//!
//! Metered execution fixes segment boundaries. Preflight then converts each
//! segment's mutable execution into immutable program and memory history.
//! Postflight derives GPU replay indexes from that history before trace
//! generation. With `rvr`, only the preflight-history producer changes: the
//! compiled executor emits checkpoints which are expanded on the GPU.

#[cfg(feature = "metrics")]
use std::time::{Duration, Instant};

#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::MeteredCtx,
    rvr::{cuda::CheckpointReplayProgram, RvrMeteredInstance},
    PreflightEndpoint, PreflightExecution, PreflightInstance, PreflightLimits,
};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{
    interpreter_preflight::PreflightInterpretedInstance, AddressSpaceHostLayout, PreflightOutput,
    VmExecutionConfig, BLOCK_FE_WIDTH,
};
use openvm_circuit::{
    arch::{
        cuda::postflight::GpuPostflightProgram, execution_mode::Segment,
        hasher::poseidon2::vm_poseidon2_hasher, ContinuationProverFn, ContinuationVmProof,
        ExitCode, GenerationError, Streams, VirtualMachineError, VmInstance, VmState,
    },
    system::memory::{merkle::public_values::UserPublicValuesProof, online::GuestMemory},
};
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
#[cfg(not(feature = "rvr"))]
use openvm_stark_backend::Val;
use openvm_stark_backend::{proof::Proof, StarkEngine};
use tracing::info_span;

#[cfg(not(feature = "rvr"))]
use crate::SdkVmConfig;
use crate::{SdkVmGpuBuilder, SC};

#[cfg(feature = "rvr")]
const CHECKPOINT_INTERVAL: usize = 512;

#[cfg(not(feature = "rvr"))]
type InterpretedPreflight =
    PreflightInterpretedInstance<Val<SC>, <SdkVmConfig as VmExecutionConfig<Val<SC>>>::Executor>;

/// Fixed-program GPU prover for standalone, independently scheduled segments.
///
/// Unlike continuation proving, each call starts a fresh preflight transcript,
/// so callers may prove segments out of order from arbitrary segment-start
/// states.
pub struct SegmentProver {
    #[cfg(feature = "rvr")]
    preflight: PreflightInstance<'static>,
    #[cfg(not(feature = "rvr"))]
    preflight: InterpretedPreflight,
    #[cfg(feature = "rvr")]
    program: CheckpointReplayProgram,
    #[cfg(not(feature = "rvr"))]
    program: GpuPostflightProgram,
}

impl SegmentProver {
    pub fn new(
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let _prepare = info_span!("prepare_preflight", group = "app_proof").entered();
        Self::prepare(instance)
    }

    fn prepare(
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

    /// Proves one segment from an arbitrary segment-start state.
    ///
    /// Final memory is returned only when the segment terminates successfully.
    pub fn prove(
        &self,
        instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<SC>, Option<GuestMemory>), VirtualMachineError> {
        let _prove_span = info_span!("total_proof").entered();
        let num_insns = segment.num_insns;

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);

        #[cfg(feature = "rvr")]
        let (state, exit_code, gpu_transcript, replay_plan) = {
            let num_insns = u32::try_from(num_insns)
                .map_err(|_| generation_error("metered instruction count exceeds u32"))?;
            let limits = PreflightLimits::new(
                num_insns as usize,
                segment.num_preflight_residuals as usize,
                CHECKPOINT_INTERVAL,
            );
            let execution = self.preflight.execute_segment(state, limits)?;
            let exit_code = matches!(&execution.endpoint, PreflightEndpoint::Terminated)
                .then_some(ExitCode::Success as u32);
            let (gpu_transcript, replay_plan) =
                SdkVmGpuBuilder::postflight(&instance.vm, &self.program, &execution, num_insns)
                    .map_err(generation_error)?;
            (execution.state, exit_code, gpu_transcript, replay_plan)
        };

        #[cfg(not(feature = "rvr"))]
        let (state, exit_code, gpu_transcript, replay_plan) = {
            let mut output =
                instance
                    .vm
                    .execute_preflight_for(&self.preflight, state, num_insns)?;
            mark_written_pages(&mut output);
            let (gpu_transcript, replay_plan) = instance
                .vm
                .postflight_history(&self.program, &output)
                .map_err(generation_error)?;
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

struct PreparedPreflight {
    #[cfg(feature = "rvr")]
    metered: RvrMeteredInstance<'static>,
    #[cfg(feature = "rvr")]
    metered_ctx: MeteredCtx,
    segment_prover: SegmentProver,
}

impl PreparedPreflight {
    fn new(
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let _prepare = info_span!("prepare_preflight", group = "app_proof").entered();
        #[cfg(feature = "rvr")]
        let exe = instance.exe();
        let segment_prover = SegmentProver::prepare(instance)?;

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
            segment_prover,
        })
    }

    fn tracegen_program(&self) -> &GpuPostflightProgram {
        self.segment_prover.tracegen_program()
    }
}

pub(crate) fn continuation_prover(
) -> ContinuationProverFn<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder> {
    let mut prepared = None;
    Box::new(move |instance, input| {
        if prepared.is_none() {
            // Publish only complete preparation so a failed attempt remains retryable.
            prepared = Some(PreparedPreflight::new(instance)?);
        }
        prove(instance, input, prepared.as_ref().unwrap())
    })
}

fn prove(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    prepared: &PreparedPreflight,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    instance.reset_state(input.clone());
    let exe = instance.exe().clone();
    let result = prove_inner(instance, input, prepared);
    if result.is_err() && instance.state().is_none() {
        // Segment execution consumes the state. Restore an allocated state so
        // this fixed-program prover remains reusable after an error.
        *instance.state_mut() = Some(instance.vm.create_initial_state(&exe, Streams::default()));
    }
    result
}

fn prove_inner(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    prepared: &PreparedPreflight,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
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
    #[cfg(feature = "metrics")]
    let mut preflight_time = Duration::ZERO;
    #[cfg(feature = "metrics")]
    let mut preflight_retired = 0u64;
    #[cfg(feature = "metrics")]
    let mut preflight_retired_by_segment = Vec::with_capacity(num_segments);
    #[cfg(all(feature = "metrics", feature = "rvr"))]
    let mut interval_count = 0u64;
    #[cfg(all(feature = "metrics", feature = "rvr"))]
    let mut residual_count = 0u64;
    #[cfg(all(feature = "metrics", feature = "rvr"))]
    let mut transcript_bytes = 0u64;

    for (segment_idx, segment) in segments.into_iter().enumerate() {
        let _segment_span = info_span!("prove_segment", segment = segment_idx).entered();
        let _prove_span = info_span!("total_proof").entered();
        let Segment {
            num_insns,
            #[cfg(feature = "rvr")]
            num_preflight_residuals,
            ..
        } = segment;

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);
        #[cfg(feature = "metrics")]
        let preflight_started = Instant::now();

        #[cfg(feature = "rvr")]
        let (next_state, gpu_transcript, replay_plan) = {
            let num_insns = u32::try_from(num_insns)
                .map_err(|_| generation_error("metered instruction count exceeds u32"))?;
            let limits = PreflightLimits::new(
                num_insns as usize,
                num_preflight_residuals as usize,
                CHECKPOINT_INTERVAL,
            );
            let execution = match reuse.take() {
                Some(transcript) => prepared
                    .segment_prover
                    .preflight
                    .execute_segment_reusing(state, limits, transcript)?,
                None => prepared
                    .segment_prover
                    .preflight
                    .execute_segment(state, limits)?,
            };
            #[cfg(feature = "metrics")]
            {
                preflight_time += preflight_started.elapsed();
                preflight_retired += u64::from(execution.retired);
                preflight_retired_by_segment.push(u64::from(execution.retired));
                interval_count += execution.transcript.checkpoints.len() as u64;
                residual_count += execution.transcript.residuals.len() as u64;
                transcript_bytes +=
                    std::mem::size_of_val(execution.transcript.checkpoints.as_slice()) as u64
                        + std::mem::size_of_val(execution.transcript.residuals.as_slice()) as u64;
            }
            validate_endpoint(
                matches!(&execution.endpoint, PreflightEndpoint::Terminated),
                segment_idx + 1 == num_segments,
            )?;
            let (gpu_transcript, replay_plan) = SdkVmGpuBuilder::postflight(
                &instance.vm,
                &prepared.segment_prover.program,
                &execution,
                num_insns,
            )
            .map_err(generation_error)?;
            let PreflightExecution {
                state: next_state,
                mut transcript,
                ..
            } = execution;
            transcript.checkpoints.clear();
            transcript.residuals.clear();
            reuse = Some(transcript);
            (next_state, gpu_transcript, replay_plan)
        };

        #[cfg(not(feature = "rvr"))]
        let (next_state, gpu_transcript, replay_plan) = {
            let mut output = instance.vm.execute_preflight_for(
                &prepared.segment_prover.preflight,
                state,
                num_insns,
            )?;
            #[cfg(feature = "metrics")]
            {
                preflight_time += preflight_started.elapsed();
                preflight_retired += num_insns;
                preflight_retired_by_segment.push(num_insns);
            }
            validate_endpoint(output.exit_code.is_some(), segment_idx + 1 == num_segments)?;
            mark_written_pages(&mut output);
            let (gpu_transcript, replay_plan) = instance
                .vm
                .postflight_history(&prepared.segment_prover.program, &output)
                .map_err(generation_error)?;
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

    #[cfg(feature = "metrics")]
    {
        let elapsed_micros = preflight_time.as_secs_f64().max(1e-9) * 1_000_000.0;
        for (segment, retired) in preflight_retired_by_segment.into_iter().enumerate() {
            metrics::counter!("execute_preflight_insns", "segment" => segment.to_string())
                .absolute(retired);
        }
        #[cfg(feature = "rvr")]
        {
            metrics::counter!("execute_preflight_intervals").absolute(interval_count);
            metrics::counter!("execute_preflight_residuals").absolute(residual_count);
            metrics::counter!("execute_preflight_transcript_bytes").absolute(transcript_bytes);
        }
        metrics::gauge!("execute_preflight_insn_mi/s")
            .set(preflight_retired as f64 / elapsed_micros);
    }

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

/// Interpreter writes bypass sparse-transfer bookkeeping. Mark pages from the
/// write events so a block that was read before its first write is included.
/// Repeated writes only repeat an idempotent page-bit update.
#[cfg(not(feature = "rvr"))]
fn mark_written_pages(output: &mut PreflightOutput) {
    let memory = &mut output.state.memory.memory;
    for write in output
        .history
        .memory
        .accesses
        .iter()
        .filter(|event| event.is_write())
    {
        let address_space = write.address_space() as usize;
        let cell_size = memory.config[address_space].layout.size();
        memory.touched_pages[address_space].mark_byte_range(
            write.pointer as usize * cell_size,
            BLOCK_FE_WIDTH * cell_size,
        );
    }
}

fn generation_error(error: impl ToString) -> VirtualMachineError {
    VirtualMachineError::Generation(GenerationError::ExtensionTracegen(error.to_string()))
}
