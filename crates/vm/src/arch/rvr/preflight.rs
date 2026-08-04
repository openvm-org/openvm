//! Compiled preflight execution.
//!
//! Serial execution still uses mutable random-access VM memory. Its
//! authoritative preflight output contains only periodic architectural
//! checkpoints and ordered replay values that deterministic replay cannot
//! recover from the program and segment-start state. GPU expansion converts
//! those arrays into a read-only logical execution history for parallel
//! tracegen.
//!
//! Dirty-page bitsets are transfer metadata, not transcript data: they identify
//! host writes that must be copied before the next segment. Proof-visible reads
//! and their predecessor timestamps are reconstructed later by GPU chronology.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use rvr_openvm_lift::RvrRuntimeExtension;
use rvr_state::RvrCheckpoint;

use super::{
    bridge::map_rvr_execute_error,
    compile::CompileError,
    execute::{commit_guest_profile, execute_preflight, PreflightExecuteOptions},
    GuestProfileConfig, RvrCompiled, RvrInitialImage,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::Segment, ExecutionError, ExecutionState, Streams, SystemConfig, VmState,
    },
    system::memory::online::GuestMemory,
};

mod buffers;

pub(crate) use buffers::{PreflightBuffers, PreflightDirtyPages};

const DEFAULT_CHECKPOINT_INTERVAL: usize = 512;

/// Why preflight stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreflightEndpoint {
    Terminated,
    Suspended,
}

/// Resource limits for one preflight execution call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreflightLimits {
    pub max_instructions: usize,
    pub max_replay_values: usize,
    pub checkpoint_interval: usize,
}

impl PreflightLimits {
    pub const fn new(
        max_instructions: usize,
        max_replay_values: usize,
        checkpoint_interval: usize,
    ) -> Self {
        Self {
            max_instructions,
            max_replay_values,
            checkpoint_interval,
        }
    }

    fn validated(self) -> Result<ValidatedLimits, String> {
        let max_instructions = u32::try_from(self.max_instructions)
            .map_err(|_| "preflight instruction limit exceeds u32".to_string())?;
        let max_replay_values_u32 = u32::try_from(self.max_replay_values).map_err(|_| {
            "preflight replay-value limit exceeds the u32 cursor stored in each checkpoint"
                .to_string()
        })?;
        let max_replay_values = u64::from(max_replay_values_u32);
        let checkpoint_interval = u32::try_from(self.checkpoint_interval)
            .map_err(|_| "preflight checkpoint interval exceeds u32".to_string())?;
        if checkpoint_interval == 0 {
            return Err("preflight checkpoint interval must be nonzero".to_string());
        }

        // Every interior checkpoint advances the instruction count by at least one
        // interval. Ceiling division remains safe when a basic block overshoots
        // an interval boundary.
        let max_checkpoints = (self.max_instructions / self.checkpoint_interval)
            .checked_add(usize::from(
                !self
                    .max_instructions
                    .is_multiple_of(self.checkpoint_interval),
            ))
            .ok_or_else(|| "preflight checkpoint bound overflow".to_string())?;
        let max_checkpoints_u64 = u64::try_from(max_checkpoints)
            .map_err(|_| "preflight checkpoint limit exceeds u64".to_string())?;

        Ok(ValidatedLimits {
            max_instructions,
            max_replay_values,
            checkpoint_interval,
            max_checkpoints,
            max_checkpoints_u64,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedLimits {
    max_instructions: u32,
    max_replay_values: u64,
    checkpoint_interval: u32,
    max_checkpoints: usize,
    max_checkpoints_u64: u64,
}

/// Minimal authoritative output of preflight.
#[derive(Debug, Default)]
pub struct PreflightTranscript {
    pub checkpoints: Vec<RvrCheckpoint>,
    pub replay_values: Vec<u64>,
}

/// State and compact transcript returned by preflight.
pub struct PreflightExecution {
    pub state: VmState<GuestMemory>,
    pub transcript: PreflightTranscript,
    pub endpoint: PreflightEndpoint,
    /// Initial execution-bus boundary. The initial register and memory image
    /// are supplied separately by the caller; they are not duplicated here.
    pub from_state: ExecutionState<u32>,
    /// Final execution-bus boundary. The final registers and memory live in
    /// `state`; they are not copied into the transcript.
    pub to_state: ExecutionState<u32>,
    pub retired: u32,
}

struct PreflightInstanceInner<'a> {
    system_config: Cow<'a, SystemConfig>,
    initial_image: RvrInitialImage,
    compiled: RvrCompiled,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

/// Compiled preflight executor.
pub struct PreflightInstance<'a> {
    inner: PreflightInstanceInner<'a>,
}

static_assertions::assert_impl_all!(PreflightInstance<'static>: Send, Sync);

impl<'a> PreflightInstance<'a> {
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            inner: PreflightInstanceInner {
                system_config: Cow::Borrowed(system_config),
                initial_image,
                compiled,
                runtime_hooks,
            },
        }
    }

    /// Detaches the compiled executor from the [`VmExecutor`](crate::arch::VmExecutor)
    /// that created it.
    pub fn into_owned(self) -> PreflightInstance<'static> {
        PreflightInstance {
            inner: PreflightInstanceInner {
                system_config: Cow::Owned(self.inner.system_config.into_owned()),
                initial_image: self.inner.initial_image,
                compiled: self.inner.compiled,
                runtime_hooks: self.inner.runtime_hooks,
            },
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner
            .initial_image
            .create_vm_state(&self.inner.system_config, inputs)
    }

    /// Whether this artifact can be passed to the profiled execution APIs.
    pub const fn is_profile_compatible(&self) -> bool {
        self.inner.compiled.is_profile_compatible()
    }

    /// Persist the compiled shared library into `dir`.
    ///
    /// Loading requires the same VM executable and execution configuration.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        let dest_lib = self.inner.compiled.artifact_file_name()?;
        self.inner.compiled.save_artifact(&dir.join(dest_lib))
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.inner.compiled.save_generated_sources(dir)
    }

    /// Low-level execution with caller-supplied transcript capacities.
    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
        limits: PreflightLimits,
    ) -> Result<PreflightExecution, ExecutionError> {
        self.execute_from_state(self.create_initial_vm_state(inputs), limits)
    }

    /// Low-level execution with caller-supplied transcript capacities.
    pub fn execute_from_state(
        &self,
        state: VmState<GuestMemory>,
        limits: PreflightLimits,
    ) -> Result<PreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, false, None, None)
    }

    /// Executes for at most `limits.max_instructions`, suspending before a
    /// basic block that would exceed the budget.
    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        limits: PreflightLimits,
    ) -> Result<PreflightExecution, ExecutionError> {
        self.execute_from_state_for(self.create_initial_vm_state(inputs), limits)
    }

    /// Continues for at most `limits.max_instructions`, suspending before a
    /// basic block that would exceed the budget.
    pub fn execute_from_state_for(
        &self,
        state: VmState<GuestMemory>,
        limits: PreflightLimits,
    ) -> Result<PreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, true, None, None)
    }

    /// Executes one metered segment from an arbitrary segment-start state.
    ///
    /// Both the instruction and replay-value counts must match the metered segment.
    pub fn execute_segment(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightExecution, ExecutionError> {
        let limits = limits_for_segment(segment)?;
        require_segment_boundary(
            self.execute_from_state_inner(state, limits, true, None, None)?,
            segment,
        )
    }

    /// Execute one metered segment while appending samples to `profile`.
    pub fn execute_segment_profiled(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
        profile: &GuestProfileConfig,
    ) -> Result<PreflightExecution, ExecutionError> {
        if !profile.is_session() {
            return Err(ExecutionError::RvrExecution(
                "profiled preflight requires GuestProfileConfig::raw_session".to_string(),
            ));
        }
        let limits = limits_for_segment(segment)?;
        let staging = profile.staging_session();
        let execution = require_segment_boundary(
            self.execute_from_state_inner(state, limits, true, None, Some(&staging))?,
            segment,
        )?;
        let capture = staging
            .take_session_profile()
            .map_err(|error| ExecutionError::RvrExecution(error.to_string()))?;
        commit_guest_profile(&self.inner.compiled, profile, capture)
            .map_err(map_rvr_execute_error)?;
        Ok(execution)
    }

    /// Allocation-reusing variant of [`Self::execute_segment`].
    pub fn execute_segment_reusing(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
        reuse: PreflightTranscript,
    ) -> Result<PreflightExecution, ExecutionError> {
        let limits = limits_for_segment(segment)?;
        require_segment_boundary(
            self.execute_from_state_inner(state, limits, true, Some(reuse), None)?,
            segment,
        )
    }

    fn execute_from_state_inner(
        &self,
        mut state: VmState<GuestMemory>,
        limits: PreflightLimits,
        allow_suspended: bool,
        reuse: Option<PreflightTranscript>,
        profile: Option<&GuestProfileConfig>,
    ) -> Result<PreflightExecution, ExecutionError> {
        let from_state = ExecutionState::new(state.pc(), 1u32);
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Preflight);
        let (transcript, endpoint, final_timestamp, retired) =
            tracing::info_span!("execute_preflight")
                .in_scope(|| {
                    execute_preflight(
                        &self.inner.compiled,
                        &self.inner.runtime_hooks,
                        &mut state,
                        PreflightExecuteOptions {
                            limits,
                            timestamp_max_bits: self
                                .inner
                                .system_config
                                .memory_config
                                .timestamp_max_bits,
                            allow_suspended,
                            reuse,
                            profile,
                        },
                    )
                })
                .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            metrics.record(u64::from(retired));
            // Interior checkpoints divide execution into one additional replay interval.
            metrics::counter!("execute_preflight_intervals")
                .absolute(transcript.checkpoints.len() as u64 + 1);
            metrics::counter!("execute_preflight_replay_values")
                .absolute(transcript.replay_values.len() as u64);
            let transcript_bytes = std::mem::size_of_val(transcript.checkpoints.as_slice()) as u64
                + std::mem::size_of_val(transcript.replay_values.as_slice()) as u64;
            metrics::counter!("execute_preflight_transcript_bytes").absolute(transcript_bytes);
        }
        let to_state = ExecutionState::new(state.pc(), final_timestamp);
        Ok(PreflightExecution {
            state,
            transcript,
            endpoint,
            from_state,
            to_state,
            retired,
        })
    }
}

fn limits_for_segment(segment: &Segment) -> Result<PreflightLimits, ExecutionError> {
    let max_instructions = usize::try_from(segment.num_insns).map_err(|_| {
        ExecutionError::RvrExecution("preflight instruction limit exceeds usize".to_string())
    })?;
    Ok(PreflightLimits::new(
        max_instructions,
        segment.num_preflight_replay_values as usize,
        DEFAULT_CHECKPOINT_INTERVAL,
    ))
}

fn require_segment_boundary(
    execution: PreflightExecution,
    segment: &Segment,
) -> Result<PreflightExecution, ExecutionError> {
    let instret = u64::from(execution.retired);
    if instret != segment.num_insns {
        return Err(ExecutionError::RetiredInstructionCountMismatch {
            expected: segment.num_insns,
            actual: instret,
        });
    }
    let replay_values = execution.transcript.replay_values.len() as u64;
    let expected_replay_values = u64::from(segment.num_preflight_replay_values);
    if replay_values != expected_replay_values {
        return Err(ExecutionError::PreflightReplayValueCountMismatch {
            expected: expected_replay_values,
            actual: replay_values,
        });
    }
    Ok(execution)
}

#[cfg(test)]
mod tests;
