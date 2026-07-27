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
        GenerationError, Streams, VirtualMachineError, VmInstance,
    },
    system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
use openvm_stark_backend::StarkEngine;
#[cfg(not(feature = "rvr"))]
use openvm_stark_backend::Val;
use tracing::info_span;

#[cfg(not(feature = "rvr"))]
use crate::SdkVmConfig;
use crate::{SdkVmGpuBuilder, SC};

#[cfg(feature = "rvr")]
const CHECKPOINT_INTERVAL: usize = 512;

#[cfg(not(feature = "rvr"))]
type InterpretedPreflight =
    PreflightInterpretedInstance<Val<SC>, <SdkVmConfig as VmExecutionConfig<Val<SC>>>::Executor>;

struct PreparedPreflight {
    #[cfg(feature = "rvr")]
    metered: RvrMeteredInstance<'static>,
    #[cfg(feature = "rvr")]
    metered_ctx: MeteredCtx,
    #[cfg(feature = "rvr")]
    preflight: PreflightInstance<'static>,
    #[cfg(not(feature = "rvr"))]
    preflight: InterpretedPreflight,
    #[cfg(feature = "rvr")]
    program: CheckpointReplayProgram,
    #[cfg(not(feature = "rvr"))]
    program: GpuPostflightProgram,
}

impl PreparedPreflight {
    fn new(
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let _prepare = info_span!("prepare_preflight", group = "app_proof").entered();
        let exe = instance.exe();

        #[cfg(feature = "rvr")]
        let (metered, metered_ctx, preflight) = {
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
            let preflight = instance
                .vm
                .executor()
                .preflight_instance(exe)
                .map_err(VirtualMachineError::from)?
                .into_owned();
            (metered, metered_ctx, preflight)
        };
        #[cfg(not(feature = "rvr"))]
        let preflight = instance.vm.preflight_interpreter(exe)?;

        let program = info_span!("upload_preflight_program")
            .in_scope(|| SdkVmGpuBuilder::upload_preflight_program(&instance.vm, &exe.program))
            .map_err(generation_error)?;

        Ok(Self {
            #[cfg(feature = "rvr")]
            metered,
            #[cfg(feature = "rvr")]
            metered_ctx,
            preflight,
            program,
        })
    }

    fn segments(
        &self,
        _instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        input: Streams,
    ) -> Result<Vec<Segment>, VirtualMachineError> {
        #[cfg(feature = "rvr")]
        {
            Ok(self
                .metered
                .execute_metered(input, self.metered_ctx.clone())?
                .0)
        }
        #[cfg(not(feature = "rvr"))]
        {
            let metered_ctx = _instance.vm.build_metered_ctx(_instance.exe());
            let metered = _instance.vm.metered_instance(_instance.exe())?;
            Ok(metered.execute_metered(input, metered_ctx)?.0)
        }
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
}

pub(crate) fn continuation_prover(
) -> ContinuationProverFn<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder> {
    let mut prepared = None;
    Box::new(move |instance, input, program_name| {
        if prepared.is_none() {
            // Publish only complete preparation so a failed attempt remains retryable.
            prepared = Some(PreparedPreflight::new(instance)?);
        }
        let _prove_span = info_span!(
            "app_prove",
            group = "app_proof",
            program = program_name.unwrap_or("")
        )
        .entered();
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
    let segments = prepared.segments(instance, input)?;
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
    let mut checkpoint_count = 0u64;
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
            num_checkpoint_residuals,
            ..
        } = segment;

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);
        #[cfg(feature = "metrics")]
        let preflight_started = Instant::now();

        #[cfg(feature = "rvr")]
        let (next_state, gpu_transcript, replay_plan) = {
            let num_insns = usize::try_from(num_insns)
                .map_err(|_| generation_error("metered instruction count exceeds usize"))?;
            let expected_retired = u32::try_from(num_insns)
                .map_err(|_| generation_error("metered instruction count exceeds u32"))?;
            let limits = PreflightLimits::new(
                num_insns,
                num_checkpoint_residuals as usize,
                CHECKPOINT_INTERVAL,
            );
            let execution = match reuse.take() {
                Some(transcript) => prepared
                    .preflight
                    .execute_segment_reusing(state, limits, transcript)?,
                None => prepared.preflight.execute_segment(state, limits)?,
            };
            #[cfg(feature = "metrics")]
            {
                preflight_time += preflight_started.elapsed();
                preflight_retired += u64::from(execution.retired);
                preflight_retired_by_segment.push(u64::from(execution.retired));
                checkpoint_count += execution.transcript.checkpoints.len() as u64;
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
                &prepared.program,
                &execution,
                expected_retired,
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
            let mut output =
                instance
                    .vm
                    .execute_preflight_for(&prepared.preflight, state, num_insns)?;
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
                .postflight_history(&prepared.program, &output)
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
            metrics::counter!("execute_preflight_checkpoints").absolute(checkpoint_count);
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
