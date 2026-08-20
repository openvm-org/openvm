//! Record-free CUDA continuation proving.
//!
//! Metered execution fixes segment boundaries. Preflight then converts each
//! segment's mutable execution into immutable program and memory history.
//! Postflight derives GPU replay indexes from that history before trace
//! generation. With `rvr`, only the preflight-history producer changes: the
//! compiled executor emits checkpoints which are expanded on the GPU.

#[cfg(feature = "rvr")]
use std::sync::{Arc, OnceLock};
#[cfg(feature = "metrics")]
use std::time::{Duration, Instant};
use std::{panic::resume_unwind, sync::Mutex};

#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::VmExecutionConfig;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::MeteredCtx,
    rvr::{cuda::CheckpointReplayProgram, RvrMeteredInstance, RvrMeteredSegmentInstance},
    PreflightEndpoint, PreflightExecution, PreflightInstance, PreflightLimits, PreflightTranscript,
};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{
    interpreter_preflight::PreflightInterpretedInstance, AddressSpaceHostLayout, PreflightOutput,
    BLOCK_FE_WIDTH,
};
use openvm_circuit::{
    arch::{
        cuda::postflight::GpuPostflightProgram, drive_scheduled, execution_mode::Segment,
        hasher::poseidon2::vm_poseidon2_hasher, ContinuationProverFn, ContinuationVmProof,
        ExitCode, GenerationError, MeteredSegmentProducer, ProvingKeyResidency, ScheduledRunRecord,
        SegmentDriver, SegmentSchedulerConfig, Streams, VirtualMachineError, VmInstance, VmState,
    },
    system::memory::{merkle::public_values::UserPublicValuesProof, online::GuestMemory},
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
#[cfg(not(feature = "rvr"))]
use openvm_stark_backend::Val;
use openvm_stark_backend::{
    proof::Proof,
    prover::{DeviceDataTransporter, DeviceMultiStarkProvingKey, ProvingContext},
    StarkEngine,
};
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
    /// Compiled on first scheduled run, so the serial path never pays for it.
    #[cfg(feature = "rvr")]
    metered_segment: OnceLock<Arc<RvrMeteredSegmentInstance<'static>>>,
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
            #[cfg(feature = "rvr")]
            metered_segment: OnceLock::new(),
            segment_prover,
        })
    }

    /// The segment-boundary executor, compiled once and reused.
    ///
    /// Streaming the segmentation needs a differently compiled executor from the
    /// one the serial path runs, and compiling it is a C compilation — far too
    /// costly to repeat per proof on the path a measurement will time.
    #[cfg(feature = "rvr")]
    fn metered_segment(
        &self,
        instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<Arc<RvrMeteredSegmentInstance<'static>>, VirtualMachineError> {
        if let Some(compiled) = self.metered_segment.get() {
            return Ok(compiled.clone());
        }
        let compiled = Arc::new(
            instance
                .vm
                .metered_segment_instance(instance.exe())?
                .into_owned(),
        );
        let _ = self.metered_segment.set(compiled.clone());
        Ok(compiled)
    }

    fn tracegen_program(&self) -> &GpuPostflightProgram {
        self.segment_prover.tracegen_program()
    }
}

/// The metered segmentation source for this build's execution mode.
#[cfg(feature = "rvr")]
type MeteredSegments = MeteredSegmentProducer;
#[cfg(not(feature = "rvr"))]
type MeteredSegments =
    MeteredSegmentProducer<<SdkVmConfig as VmExecutionConfig<Val<SC>>>::Executor>;

/// One admitted prove's engine, and the proving key its stream reads.
struct ProveSlot {
    engine: BabyBearPoseidon2GpuEngine,
    /// This slot's own key, or `None` to read the key the VM already holds.
    pk: Option<DeviceMultiStarkProvingKey<GpuBackend>>,
}

impl ProveSlot {
    /// Identifies the CUDA stream this slot enqueues on.
    ///
    /// Two slots proving at the same time must not report the same value. Equal
    /// handles mean one stream, and one stream means the proves' kernels run in
    /// issue order rather than together — while still producing correct proofs,
    /// so nothing but this observation distinguishes the two.
    fn queue_id(&self) -> u64 {
        self.engine.device().device_ctx.stream.as_raw() as u64
    }

    fn pk<'a>(
        &'a self,
        shared: &'a DeviceMultiStarkProvingKey<GpuBackend>,
    ) -> &'a DeviceMultiStarkProvingKey<GpuBackend> {
        self.pk.as_ref().unwrap_or(shared)
    }
}

/// Builds one engine per prove the budget admits, each owning a fresh CUDA stream.
///
/// [`BabyBearPoseidon2GpuEngine::new`] mints a new device context, and each device
/// context creates its own stream. Cloning one engine would instead share its
/// `Arc<CudaStream>`, so every prove would enqueue on a single stream and their
/// kernel work would serialize — correct, and invisible to every correctness
/// check we have. Separate engines are what make the concurrency real.
fn build_prove_pool(
    instance: &VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    slots: usize,
) -> Vec<ProveSlot> {
    let residency = instance.prove_pk_residency().clone();
    let params = instance.vm.engine.params().clone();
    (0..slots)
        .map(|_| {
            let engine = BabyBearPoseidon2GpuEngine::new(params.clone());
            let pk = match &residency {
                ProvingKeyResidency::Shared => None,
                // Every slot transports its own, including the first: a slot
                // reading a key uploaded on someone else's stream is a cross-stream
                // read, which is the dependency this shape exists to avoid. The VM
                // keeps its own key besides these, so N slots are N + 1 resident.
                ProvingKeyResidency::PerProve(host_pk) => {
                    Some(engine.device().transport_pk_to_device(host_pk))
                }
            };
            ProveSlot { engine, pk }
        })
        .collect()
}

/// Preflight metric accumulators, reported once per run.
#[cfg(feature = "metrics")]
#[derive(Default)]
struct PreflightMetrics {
    time: Duration,
    retired: u64,
    #[cfg(feature = "rvr")]
    retired_by_segment: Vec<u64>,
    #[cfg(feature = "rvr")]
    intervals: u64,
    #[cfg(feature = "rvr")]
    residuals: u64,
    #[cfg(feature = "rvr")]
    transcript_bytes: u64,
}

#[cfg(feature = "metrics")]
impl PreflightMetrics {
    fn emit(self) {
        let elapsed_micros = self.time.as_secs_f64().max(1e-9) * 1_000_000.0;
        #[cfg(feature = "rvr")]
        {
            for (segment, retired) in self.retired_by_segment.into_iter().enumerate() {
                metrics::counter!("execute_preflight_insns", "segment" => segment.to_string())
                    .absolute(retired);
            }
            metrics::counter!("execute_preflight_intervals").absolute(self.intervals);
            metrics::counter!("execute_preflight_residuals").absolute(self.residuals);
            metrics::counter!("execute_preflight_transcript_bytes").absolute(self.transcript_bytes);
        }
        metrics::gauge!("execute_preflight_insn_mi/s").set(self.retired as f64 / elapsed_micros);
    }
}

/// Advances the execute chain one segment at a time: preflight, postflight, then
/// trace generation, ending with a proving context ready to be proved.
///
/// Segment `n + 1`'s preflight starts from segment `n`'s output state, so this is
/// inherently serial and both drivers step it the same way.
struct SegmentExecutor<'a> {
    instance: &'a mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    prepared: &'a PreparedPreflight,
    state: Option<VmState<GuestMemory>>,
    #[cfg(feature = "rvr")]
    reuse: Option<PreflightTranscript>,
    /// Whether each executed segment terminated, in segment order.
    terminated: Vec<bool>,
    #[cfg(feature = "metrics")]
    metrics: PreflightMetrics,
}

impl<'a> SegmentExecutor<'a> {
    fn new(
        instance: &'a mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        prepared: &'a PreparedPreflight,
    ) -> Result<Self, VirtualMachineError> {
        let state = instance
            .state_mut()
            .take()
            .ok_or_else(|| generation_error("VM instance has no execution state"))?;
        Ok(Self {
            instance,
            prepared,
            state: Some(state),
            #[cfg(feature = "rvr")]
            reuse: None,
            terminated: Vec::new(),
            #[cfg(feature = "metrics")]
            metrics: PreflightMetrics::default(),
        })
    }

    fn execute(
        &mut self,
        segment: &Segment,
    ) -> Result<ProvingContext<GpuBackend>, VirtualMachineError> {
        let num_insns = segment.num_insns;
        #[cfg(feature = "rvr")]
        let num_preflight_residuals = segment.num_preflight_residuals;
        // A shared reference field, so copying it out leaves `self` free to be
        // borrowed mutably alongside it.
        let prepared = self.prepared;
        let state = self
            .state
            .take()
            .expect("the execute chain carries the state forward");

        // Replay resolves first reads against the immutable segment-start
        // image, so upload it before preflight mutates host memory.
        self.instance
            .vm
            .transport_init_memory_to_device(&state.memory);
        #[cfg(feature = "metrics")]
        let preflight_started = Instant::now();

        #[cfg(feature = "rvr")]
        let (next_state, gpu_transcript, replay_plan, terminated) = {
            let num_insns = u32::try_from(num_insns)
                .map_err(|_| generation_error("metered instruction count exceeds u32"))?;
            let limits = PreflightLimits::new(
                num_insns as usize,
                num_preflight_residuals as usize,
                CHECKPOINT_INTERVAL,
            );
            let execution = match self.reuse.take() {
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
                self.metrics.time += preflight_started.elapsed();
                self.metrics.retired += u64::from(execution.retired);
                self.metrics
                    .retired_by_segment
                    .push(u64::from(execution.retired));
                self.metrics.intervals += execution.transcript.checkpoints.len() as u64;
                self.metrics.residuals += execution.transcript.residuals.len() as u64;
                self.metrics.transcript_bytes +=
                    std::mem::size_of_val(execution.transcript.checkpoints.as_slice()) as u64
                        + std::mem::size_of_val(execution.transcript.residuals.as_slice()) as u64;
            }
            let terminated = matches!(&execution.endpoint, PreflightEndpoint::Terminated);
            let (gpu_transcript, replay_plan) = SdkVmGpuBuilder::postflight(
                &self.instance.vm,
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
            self.reuse = Some(transcript);
            (next_state, gpu_transcript, replay_plan, terminated)
        };

        #[cfg(not(feature = "rvr"))]
        let (next_state, gpu_transcript, replay_plan, terminated) = {
            let mut output = self.instance.vm.execute_preflight_for(
                &prepared.segment_prover.preflight,
                state,
                num_insns,
            )?;
            #[cfg(feature = "metrics")]
            {
                self.metrics.time += preflight_started.elapsed();
                self.metrics.retired += num_insns;
            }
            let terminated = output.exit_code.is_some();
            mark_written_pages(&mut output);
            let (gpu_transcript, replay_plan) = self
                .instance
                .vm
                .postflight_history(&prepared.segment_prover.program, &output)
                .map_err(generation_error)?;
            (output.state, gpu_transcript, replay_plan, terminated)
        };

        self.state = Some(next_state);
        self.terminated.push(terminated);
        let ctx = SdkVmGpuBuilder::generate_preflight_proving_ctx(
            &mut self.instance.vm,
            prepared.tracegen_program(),
            &gpu_transcript,
            &replay_plan,
        )?;

        // Tracegen has synchronized before returning. Release segment-sized
        // replay data before the proving phase reaches its memory peak.
        drop(replay_plan);
        drop(gpu_transcript);
        Ok(ctx)
    }

    /// Errors unless exactly the last segment terminated.
    ///
    /// Checked once the run is complete rather than per segment: streamed
    /// segmentation does not know the segment count until the last boundary, and
    /// the two forms constrain the same set of runs.
    fn validate_endpoints(&self) -> Result<(), VirtualMachineError> {
        let last = self.terminated.len().saturating_sub(1);
        for (idx, terminated) in self.terminated.iter().enumerate() {
            validate_endpoint(*terminated, idx == last)?;
        }
        Ok(())
    }

    fn take_state(&mut self) -> Result<VmState<GuestMemory>, VirtualMachineError> {
        self.state
            .take()
            .ok_or_else(|| generation_error("the execute chain produced no final state"))
    }
}

/// Drives the record-free CUDA path against the scheduler graph.
struct GpuSegmentDriver<'a> {
    executor: SegmentExecutor<'a>,
    pool: &'a [ProveSlot],
    /// Per dispatched batch, each segment index with the stream that proved it.
    queues: Mutex<Vec<Vec<(usize, u64)>>>,
}

impl SegmentDriver for GpuSegmentDriver<'_> {
    type Ctx = ProvingContext<GpuBackend>;
    type Proof = Proof<SC>;

    fn execute(&mut self, idx: usize, segment: &Segment) -> Result<Self::Ctx, VirtualMachineError> {
        let _span = info_span!("execute_segment", segment = idx).entered();
        self.executor.execute(segment)
    }

    fn prove_batch(
        &self,
        batch: Vec<(usize, Self::Ctx)>,
        while_proving: &mut dyn FnMut() -> Result<Vec<Segment>, VirtualMachineError>,
    ) -> Result<(Vec<(usize, Self::Proof)>, Vec<Segment>), VirtualMachineError> {
        assert!(
            batch.len() <= self.pool.len(),
            "admission let in {} proves but only {} streams exist to run them on",
            batch.len(),
            self.pool.len()
        );
        let shared_pk = self.executor.instance.vm.pk();
        self.queues
            .lock()
            .expect("prove queues are recorded without panicking")
            .push(
                batch
                    .iter()
                    .enumerate()
                    .map(|(slot, (idx, _))| (*idx, self.pool[slot].queue_id()))
                    .collect(),
            );
        let (results, produced) = std::thread::scope(|scope| {
            let handles = batch
                .into_iter()
                .enumerate()
                .map(|(slot, (idx, ctx))| {
                    // One slot per batch position, so no two proves in flight
                    // share a stream.
                    let slot = &self.pool[slot];
                    scope.spawn(move || {
                        (
                            idx,
                            slot.engine
                                .prove(slot.pk(shared_pk), ctx)
                                .map_err(|error| error.to_string()),
                        )
                    })
                })
                .collect::<Vec<_>>();
            // The proves are in flight and borrow nothing this thread needs, so
            // segment production runs here rather than after the join. This is the
            // producer/prove overlap.
            let produced = while_proving();
            let results = handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|payload| resume_unwind(payload))
                })
                .collect::<Vec<_>>();
            (results, produced)
        });
        let proofs = results
            .into_iter()
            .map(|(idx, proof)| match proof {
                Ok(proof) => Ok((idx, proof)),
                Err(error) => Err(VirtualMachineError::Generation(GenerationError::Proving(
                    error,
                ))),
            })
            .collect::<Result<Vec<_>, VirtualMachineError>>()?;
        // Reported only after the join, so a producer failure never leaves proves
        // running unobserved.
        Ok((proofs, produced?))
    }
}

pub(crate) fn continuation_prover(
) -> ContinuationProverFn<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder> {
    let mut prepared = None;
    let mut pool: Vec<ProveSlot> = Vec::new();
    let mut pool_shape: Option<(usize, bool)> = None;
    Box::new(move |instance, input| {
        if prepared.is_none() {
            // Publish only complete preparation so a failed attempt remains retryable.
            prepared = Some(PreparedPreflight::new(instance)?);
        }
        let scheduler = instance.segment_scheduler();
        if let Some(scheduler) = &scheduler {
            let shape = (
                scheduler.max_resident_proves(),
                matches!(
                    instance.prove_pk_residency(),
                    ProvingKeyResidency::PerProve(_)
                ),
            );
            if pool_shape != Some(shape) {
                // Released before the replacement is built, so two pools of device
                // proving keys are never resident at once.
                drop(std::mem::take(&mut pool));
                pool = build_prove_pool(instance, shape.0);
                pool_shape = Some(shape);
            }
        }
        prove(
            instance,
            input,
            prepared.as_ref().unwrap(),
            &pool,
            scheduler.as_ref(),
        )
    })
}

fn prove(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    prepared: &PreparedPreflight,
    pool: &[ProveSlot],
    scheduler: Option<&SegmentSchedulerConfig>,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    instance.reset_state(input.clone());
    let exe = instance.exe().clone();
    let result = match scheduler {
        None => prove_serial(instance, input, prepared),
        Some(scheduler) => prove_scheduled(instance, input, prepared, pool, scheduler),
    };
    if result.is_err() && instance.state().is_none() {
        // Segment execution consumes the state. Restore an allocated state so
        // this fixed-program prover remains reusable after an error.
        *instance.state_mut() = Some(instance.vm.create_initial_state(&exe, Streams::default()));
    }
    result
}

fn boundaries(segments: &[Segment]) -> Vec<(u64, u64)> {
    segments
        .iter()
        .map(|segment| (segment.instret_start, segment.num_insns))
        .collect()
}

/// Proves one segment at a time, each prove finishing before the next execute.
fn prove_serial(
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
    let mut proofs = Vec::with_capacity(segments.len());
    let mut executor = SegmentExecutor::new(&mut *instance, prepared)?;
    for (segment_idx, segment) in segments.iter().enumerate() {
        let _segment_span = info_span!("prove_segment", segment = segment_idx).entered();
        // A separate span so the metric label includes "segment" from _segment_span
        let _prove_span = info_span!("total_proof").entered();
        let ctx = executor.execute(segment)?;
        let proof = executor
            .instance
            .vm
            .engine
            .prove(executor.instance.vm.pk(), ctx)
            .map_err(|error| GenerationError::Proving(error.to_string()))?;
        proofs.push(proof);
    }
    executor.validate_endpoints()?;
    let state = executor.take_state()?;
    #[cfg(feature = "metrics")]
    let metrics = std::mem::take(&mut executor.metrics);
    drop(executor);

    instance.set_scheduled_run(ScheduledRunRecord {
        segment_boundaries: boundaries(&segments),
        ..Default::default()
    });
    #[cfg(feature = "metrics")]
    metrics.emit();
    finish_run(instance, proofs, state)
}

/// Proves the same segments as [`prove_serial`], but lets the scheduler graph
/// decide when each half of a segment runs, so proves can be in flight together.
fn prove_scheduled(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    input: Streams,
    prepared: &PreparedPreflight,
    pool: &[ProveSlot],
    scheduler: &SegmentSchedulerConfig,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
    let exe = instance.exe().clone();
    #[cfg(feature = "rvr")]
    let mut source = MeteredSegments::from_compiled(
        prepared.metered_segment(instance)?,
        instance.vm.build_metered_ctx(&exe),
        input,
    );
    #[cfg(not(feature = "rvr"))]
    let mut source = MeteredSegments::new(&instance.vm, &exe, input)?;
    let mut driver = GpuSegmentDriver {
        executor: SegmentExecutor::new(&mut *instance, prepared)?,
        pool,
        queues: Mutex::new(Vec::new()),
    };
    let run = drive_scheduled(scheduler, &mut driver, &mut source)?;
    source.finish()?;

    let GpuSegmentDriver {
        mut executor,
        queues,
        ..
    } = driver;
    executor.validate_endpoints()?;
    let state = executor.take_state()?;
    #[cfg(feature = "metrics")]
    let metrics = std::mem::take(&mut executor.metrics);
    drop(executor);
    let prove_batch_queues = queues
        .into_inner()
        .expect("prove queues are recorded without panicking");

    instance.set_scheduled_run(ScheduledRunRecord {
        max_concurrent_proves: run.max_concurrent_proves,
        segment_boundaries: boundaries(&run.segments),
        prove_batch_queues,
    });
    #[cfg(feature = "metrics")]
    metrics.emit();
    finish_run(instance, run.proofs, state)
}

/// Computes the public values proof and hands the final state back to `instance`.
fn finish_run(
    instance: &mut VmInstance<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    per_segment: Vec<Proof<SC>>,
    state: VmState<GuestMemory>,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
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
        per_segment,
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use openvm_stark_sdk::config::app_params_with_100_bits_security;

    use super::*;

    /// Nothing is proved here, so the stacked height only has to be legal.
    const LOG_STACKED_HEIGHT: usize = 20;

    fn stream_of(engine: &BabyBearPoseidon2GpuEngine) -> u64 {
        engine.device().device_ctx.stream.as_raw() as u64
    }

    /// The distinction the prove pool rests on.
    ///
    /// A cloned device shares its `Arc<CudaStream>`, so proves dispatched against
    /// clones enqueue on one stream and their kernel work runs in issue order
    /// rather than together — while still producing correct proofs, which is why
    /// no proof-level check can see it. Separately constructed engines do not
    /// share a stream, and that is what the pool relies on.
    #[test]
    fn separate_engines_own_separate_streams_and_a_clone_does_not() {
        let params = app_params_with_100_bits_security(LOG_STACKED_HEIGHT);
        let first = BabyBearPoseidon2GpuEngine::new(params.clone());
        let second = BabyBearPoseidon2GpuEngine::new(params);
        assert_ne!(
            stream_of(&first),
            stream_of(&second),
            "separately constructed engines must not share a CUDA stream"
        );

        let cloned = first.device().clone();
        assert_eq!(
            cloned.device_ctx.stream.as_raw() as u64,
            stream_of(&first),
            "a cloned device shares the stream it was cloned from"
        );
    }
}
