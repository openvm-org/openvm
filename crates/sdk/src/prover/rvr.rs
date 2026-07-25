//! Record-free GPU proving from RVR checkpoint execution.
//!
//! ```text
//! metered RVR -> exact segment boundaries
//!                         |
//! segment-start memory ---+--> checkpoint RVR -> final mutable VM state
//!                                      |
//!                              checkpoints + residuals
//!                                      |
//!                         GPU count/emit expansion
//!                                      |
//!                       chronology + opcode indexes
//!                                      |
//!                     immutable read-only replay view
//!                         |          |          |
//!                       system     RV64     extensions
//!                                      |
//!                         drop replay scratch -> prove
//! ```
//!
//! Segment-start memory is uploaded before checkpoint execution mutates the
//! host state. The checkpoint log seeds parallel replay; the derived program
//! and memory logs are not executor output and do not survive into the proving
//! memory peak.

#[cfg(feature = "metrics")]
use std::time::{Duration, Instant};

use openvm_circuit::{
    arch::{
        execution_mode::{MeteredCtx, Segment},
        hasher::poseidon2::vm_poseidon2_hasher,
        rvr::{
            cuda::GpuRvrProgram, RvrCheckpointPreflightExecution, RvrCheckpointPreflightInstance,
            RvrCheckpointPreflightLimits, RvrMeteredInstance, RvrPreflightEndpoint,
        },
        ContinuationVmProof, ExecutionError, GenerationError, Streams, VirtualMachineError,
        VmInstance,
    },
    system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
use openvm_sdk_config::SdkVmGpuBuilder;
use openvm_stark_backend::StarkEngine;
use tracing::info_span;

use crate::{StdIn, SC};

const CHECKPOINT_INTERVAL: usize = 512;

/// Program-specific executors and device data prepared once and reused by
/// every checkpoint proof.
pub(super) struct RvrCheckpointRuntime<'a> {
    metered: RvrMeteredInstance<'a>,
    metered_ctx: MeteredCtx,
    checkpoint: RvrCheckpointPreflightInstance<'a>,
    gpu_program: GpuRvrProgram,
}

impl<'a> RvrCheckpointRuntime<'a> {
    pub(crate) fn new(
        metered: RvrMeteredInstance<'a>,
        metered_ctx: MeteredCtx,
        checkpoint: RvrCheckpointPreflightInstance<'a>,
        gpu_program: GpuRvrProgram,
    ) -> Self {
        Self {
            metered,
            metered_ctx,
            checkpoint,
            gpu_program,
        }
    }
}

/// Explicit continuation driver for compact RVR checkpoint preflight and
/// record-free GPU trace generation.
pub(super) fn prove(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: StdIn,
    runtime: &RvrCheckpointRuntime<'_>,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    let input: Streams = input.into();
    instance.reset_state(input.clone());
    let exe = instance.exe().clone();

    let result = prove_inner(instance, input, runtime);
    if result.is_err() && instance.state().is_none() {
        // Checkpoint execution consumes the segment state. If it fails before
        // returning that state, restore an allocated initial state so the
        // fixed-program instance remains reusable. The next proof resets its
        // memory and streams before execution.
        *instance.state_mut() = Some(instance.vm.create_initial_state(&exe, Streams::default()));
    }
    result
}

fn prove_inner(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    runtime: &RvrCheckpointRuntime<'_>,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    // Meter once. Its exact instruction and residual counts are the only
    // capacities supplied to checkpoint preflight for each segment.
    let (segments, _) = runtime
        .metered
        .execute_metered(input, runtime.metered_ctx.clone())?;

    let num_segments = segments.len();
    let mut proofs = Vec::with_capacity(num_segments);
    let mut state = instance
        .state_mut()
        .take()
        .ok_or_else(|| execution_error("VM instance has no execution state"))?;
    let mut reuse = None;
    #[cfg(feature = "metrics")]
    let mut checkpoint_execution_time = Duration::ZERO;
    #[cfg(feature = "metrics")]
    let mut checkpoint_retired = 0u64;
    #[cfg(feature = "metrics")]
    let mut checkpoint_retired_by_segment = Vec::with_capacity(num_segments);
    #[cfg(feature = "metrics")]
    let mut checkpoint_count = 0u64;
    #[cfg(feature = "metrics")]
    let mut residual_count = 0u64;
    #[cfg(feature = "metrics")]
    let mut transcript_bytes = 0u64;

    for (segment_idx, segment) in segments.into_iter().enumerate() {
        let _segment_span = info_span!("prove_segment", segment = segment_idx).entered();
        let _prove_span = info_span!("total_proof").entered();
        let Segment {
            num_insns,
            num_checkpoint_residuals,
            ..
        } = segment;
        let num_insns = usize::try_from(num_insns)
            .map_err(|_| execution_error("metered segment instruction count exceeds usize"))?;
        let expected_retired = u32::try_from(num_insns)
            .map_err(|_| execution_error("metered segment instruction count exceeds u32"))?;
        let limits = RvrCheckpointPreflightLimits::new(
            num_insns,
            num_checkpoint_residuals as usize,
            CHECKPOINT_INTERVAL,
        );

        // Replay resolves its first reads against this immutable segment-start
        // image, so upload it before checkpoint execution mutates host memory.
        instance.vm.transport_init_memory_to_device(&state.memory);
        #[cfg(feature = "metrics")]
        let checkpoint_execution_started = Instant::now();
        let execution = match reuse.take() {
            Some(transcript) => runtime
                .checkpoint
                .execute_from_state_for_exact_reusing(state, limits, transcript)?,
            None => runtime
                .checkpoint
                .execute_from_state_for_exact(state, limits)?,
        };
        #[cfg(feature = "metrics")]
        {
            checkpoint_execution_time += checkpoint_execution_started.elapsed();
            checkpoint_retired += u64::from(execution.retired);
            checkpoint_retired_by_segment.push(u64::from(execution.retired));
            checkpoint_count += execution.transcript.checkpoints.len() as u64;
            residual_count += execution.transcript.residuals.len() as u64;
            transcript_bytes += std::mem::size_of_val(execution.transcript.checkpoints.as_slice())
                as u64
                + std::mem::size_of_val(execution.transcript.residuals.as_slice()) as u64;
        }
        validate_endpoint(&execution, segment_idx + 1 == num_segments)?;

        let (gpu_transcript, replay_plan) = SdkVmGpuBuilder::expand_checkpoint_replay(
            &instance.vm,
            &runtime.gpu_program,
            &execution,
            expected_retired,
        )
        .map_err(generation_error)?;
        let ctx = SdkVmGpuBuilder::generate_proving_ctx_from_rvr(
            &mut instance.vm,
            &runtime.gpu_program,
            &gpu_transcript,
            &replay_plan,
        )?;

        let RvrCheckpointPreflightExecution {
            state: next_state,
            mut transcript,
            ..
        } = execution;
        state = next_state;
        transcript.checkpoints.clear();
        transcript.residuals.clear();
        reuse = Some(transcript);

        // Tracegen kernels have synchronized before returning. Release their
        // segment-sized logs and replay indexes before the proving phase peaks.
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

    // Only the proof driver knows the complete segment set. Emit after all
    // proof outputs and final VM state have been produced successfully so a
    // failed proof never leaves a partial segment series.
    #[cfg(feature = "metrics")]
    {
        let elapsed_micros = checkpoint_execution_time.as_secs_f64().max(1e-9) * 1_000_000.0;
        for (segment, retired) in checkpoint_retired_by_segment.into_iter().enumerate() {
            metrics::counter!(
                "execute_checkpoint_preflight_insns",
                "segment" => segment.to_string()
            )
            .absolute(retired);
        }
        metrics::counter!("execute_checkpoint_preflight_checkpoints").absolute(checkpoint_count);
        metrics::counter!("execute_checkpoint_preflight_residuals").absolute(residual_count);
        metrics::counter!("execute_checkpoint_preflight_transcript_bytes")
            .absolute(transcript_bytes);
        metrics::gauge!("execute_checkpoint_preflight_insn_mi/s")
            .set(checkpoint_retired as f64 / elapsed_micros);
    }

    Ok(ContinuationVmProof {
        per_segment: proofs,
        user_public_values,
    })
}

fn validate_endpoint(
    execution: &RvrCheckpointPreflightExecution,
    is_final_segment: bool,
) -> Result<(), VirtualMachineError> {
    let valid = matches!(
        (&execution.endpoint, is_final_segment),
        (RvrPreflightEndpoint::Suspended { .. }, false) | (RvrPreflightEndpoint::Terminated, true)
    );
    if valid {
        Ok(())
    } else {
        Err(execution_error(if is_final_segment {
            "final metered segment suspended instead of terminating"
        } else {
            "non-final metered segment terminated"
        }))
    }
}

fn execution_error(message: impl Into<String>) -> VirtualMachineError {
    VirtualMachineError::Execution(ExecutionError::RvrExecution(message.into()))
}

fn generation_error(error: impl ToString) -> VirtualMachineError {
    VirtualMachineError::Generation(GenerationError::ExtensionTracegen(error.to_string()))
}
