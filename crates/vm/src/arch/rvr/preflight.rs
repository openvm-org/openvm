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

use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use rvr_openvm_lift::RvrRuntimeExtension;
use rvr_state::{PreflightTranscriptState, RvrCheckpoint, PREFLIGHT_DIRTY_PAGE_BYTES};

use super::{
    bridge::map_rvr_execute_error, compile::CompileError, execute::execute_preflight, RvrCompiled,
    RvrInitialImage,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::Segment, ExecutionError, ExecutionState, Streams, SystemConfig, VmState,
    },
    system::memory::{
        online::{GuestMemory, LinearMemory, PAGE_SIZE},
        AddressMap,
    },
};

const _: () = assert!(PREFLIGHT_DIRTY_PAGE_BYTES == PAGE_SIZE);
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

pub(crate) struct PreflightBuffers {
    checkpoints: Vec<RvrCheckpoint>,
    replay_values: Vec<u64>,
    limits: ValidatedLimits,
}

/// Executor-only sparse-upload metadata. This is intentionally absent from
/// [`PreflightTranscript`]: replay does not consume it.
pub(crate) struct PreflightDirtyPages {
    memory: Box<[u64]>,
    public_values: Box<[u64]>,
    deferral: Box<[u64]>,
}

impl PreflightDirtyPages {
    pub(crate) fn new(memory: &AddressMap) -> Result<Self, String> {
        Ok(Self {
            memory: zeroed_dirty_page_words(memory.mem[RV64_MEMORY_AS as usize].size())?,
            public_values: zeroed_dirty_page_words(memory.mem[PUBLIC_VALUES_AS as usize].size())?,
            deferral: zeroed_dirty_page_words(memory.mem[DEFERRAL_AS as usize].size())?,
        })
    }

    pub(crate) fn deferral_mut(&mut self) -> &mut [u64] {
        &mut self.deferral
    }

    pub(crate) fn merge_into(&self, memory: &mut AddressMap) {
        merge_dirty_page_words(memory, RV64_MEMORY_AS, &self.memory);
        merge_dirty_page_words(memory, PUBLIC_VALUES_AS, &self.public_values);
        merge_dirty_page_words(memory, DEFERRAL_AS, &self.deferral);

        // Generated execution keeps registers in RvState and copies them back
        // only after a successful execution boundary. Mark their single page
        // when preflight finalizes successfully.
        if memory.mem[RV64_REGISTER_AS as usize].size() != 0 {
            memory.touched_pages[RV64_REGISTER_AS as usize].mark_byte_range(0, 1);
        }
    }
}

fn zeroed_dirty_page_words(num_bytes: usize) -> Result<Box<[u64]>, String> {
    let num_pages = num_bytes.div_ceil(PAGE_SIZE);
    let num_words = num_pages.div_ceil(u64::BITS as usize);
    let mut words = Vec::new();
    words
        .try_reserve_exact(num_words)
        .map_err(|error| format!("failed to reserve preflight dirty-page bits: {error}"))?;
    words.resize(num_words, 0);
    Ok(words.into_boxed_slice())
}

fn merge_dirty_page_words(memory: &mut AddressMap, address_space: u32, words: &[u64]) {
    let num_bytes = memory.mem[address_space as usize].size();
    for (word_index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let page = word_index * u64::BITS as usize + bit;
            let byte_start = page * PAGE_SIZE;
            debug_assert!(byte_start < num_bytes);
            memory.touched_pages[address_space as usize].mark_byte_range(byte_start, 1);
            remaining &= remaining - 1;
        }
    }
}

impl PreflightBuffers {
    pub(crate) fn new(limits: PreflightLimits) -> Result<Self, String> {
        Self::with_transcript(limits, PreflightTranscript::default())
    }

    pub(crate) fn reuse(
        limits: PreflightLimits,
        transcript: PreflightTranscript,
    ) -> Result<Self, String> {
        Self::with_transcript(limits, transcript)
    }

    fn with_transcript(
        limits: PreflightLimits,
        mut transcript: PreflightTranscript,
    ) -> Result<Self, String> {
        let limits = limits.validated()?;
        transcript.checkpoints.clear();
        transcript.replay_values.clear();
        if transcript.checkpoints.capacity() < limits.max_checkpoints {
            transcript
                .checkpoints
                .try_reserve_exact(limits.max_checkpoints)
                .map_err(|error| format!("failed to reserve preflight checkpoints: {error}"))?;
        }
        let max_replay_values = usize::try_from(limits.max_replay_values)
            .map_err(|_| "preflight replay-value limit exceeds usize".to_string())?;
        if transcript.replay_values.capacity() < max_replay_values {
            transcript
                .replay_values
                .try_reserve_exact(max_replay_values)
                .map_err(|error| format!("failed to reserve preflight replay values: {error}"))?;
        }
        Ok(Self {
            checkpoints: transcript.checkpoints,
            replay_values: transcript.replay_values,
            limits,
        })
    }

    pub(crate) fn ffi_state(
        &mut self,
        dirty_pages: &mut PreflightDirtyPages,
    ) -> PreflightTranscriptState {
        PreflightTranscriptState {
            checkpoint_log: self.checkpoints.as_mut_ptr(),
            replay_value_log: self.replay_values.as_mut_ptr(),
            checkpoint_log_len: 0,
            checkpoint_log_cap: self.limits.max_checkpoints_u64,
            replay_value_log_len: 0,
            replay_value_log_cap: self.limits.max_replay_values,
            timestamp: 1,
            retired: 0,
            checkpoint_interval: self.limits.checkpoint_interval,
            last_checkpoint_retired: 0,
            error: 0,
            instruction_limit: self.limits.max_instructions,
            memory_dirty_pages: dirty_pages.memory.as_mut_ptr(),
            public_values_dirty_pages: dirty_pages.public_values.as_mut_ptr(),
            memory_dirty_page_words: dirty_pages.memory.len() as u64,
            public_values_dirty_page_words: dirty_pages.public_values.len() as u64,
            last_memory_dirty_page: u32::MAX,
            padding: 0,
        }
    }

    /// # Safety
    ///
    /// `ffi` must be the state returned by [`Self::ffi_state`], and generated
    /// execution must not have changed either allocation or its capacity.
    pub(crate) unsafe fn finish(
        mut self,
        ffi: &PreflightTranscriptState,
        timestamp_max_bits: usize,
        dirty_pages: &PreflightDirtyPages,
    ) -> Result<(PreflightTranscript, u32, u32), String> {
        let (checkpoint_len, replay_value_len) =
            self.validate_ffi_state(ffi, timestamp_max_bits, dirty_pages)?;

        // SAFETY: the generated logger initialized exactly these prefixes and
        // the returned lengths were checked against the original capacities.
        unsafe {
            self.checkpoints.set_len(checkpoint_len);
            self.replay_values.set_len(replay_value_len);
        }
        validate_checkpoints(
            &self.checkpoints,
            ffi.timestamp,
            ffi.retired,
            replay_value_len,
            self.limits.checkpoint_interval,
            ffi.last_checkpoint_retired,
        )?;

        Ok((
            PreflightTranscript {
                checkpoints: self.checkpoints,
                replay_values: self.replay_values,
            },
            ffi.timestamp,
            ffi.retired,
        ))
    }

    fn validate_ffi_state(
        &self,
        ffi: &PreflightTranscriptState,
        timestamp_max_bits: usize,
        dirty_pages: &PreflightDirtyPages,
    ) -> Result<(usize, usize), String> {
        if ffi.error != 0 {
            return Err(format!(
                "generated preflight logger failed with code {}",
                ffi.error
            ));
        }
        self.validate_preserved_ffi_inputs(ffi, dirty_pages)?;

        let checkpoint_len = usize::try_from(ffi.checkpoint_log_len)
            .map_err(|_| "preflight checkpoint length exceeds usize".to_string())?;
        let replay_value_len = usize::try_from(ffi.replay_value_log_len)
            .map_err(|_| "preflight replay-value length exceeds usize".to_string())?;
        if checkpoint_len > self.limits.max_checkpoints
            || checkpoint_len > self.checkpoints.capacity()
            || replay_value_len > self.replay_values.capacity()
            || ffi.replay_value_log_len > self.limits.max_replay_values
        {
            return Err("generated preflight logger returned an out-of-bounds length".to_string());
        }
        if ffi.retired > self.limits.max_instructions {
            return Err(format!(
                "preflight retired {} instructions beyond its {} instruction limit",
                ffi.retired, self.limits.max_instructions
            ));
        }

        let timestamp_limit = 1u32
            .checked_shl(
                u32::try_from(timestamp_max_bits)
                    .map_err(|_| "preflight timestamp width does not fit u32".to_string())?,
            )
            .ok_or_else(|| "preflight timestamp width must be less than 32".to_string())?;
        if ffi.timestamp >= timestamp_limit {
            return Err(format!(
                "preflight final timestamp {} is outside the configured {timestamp_max_bits}-bit domain",
                ffi.timestamp
            ));
        }

        Ok((checkpoint_len, replay_value_len))
    }

    fn validate_preserved_ffi_inputs(
        &self,
        ffi: &PreflightTranscriptState,
        dirty_pages: &PreflightDirtyPages,
    ) -> Result<(), String> {
        if ffi.checkpoint_log != self.checkpoints.as_ptr().cast_mut()
            || ffi.replay_value_log != self.replay_values.as_ptr().cast_mut()
            || ffi.checkpoint_log_cap != self.limits.max_checkpoints_u64
            || ffi.replay_value_log_cap != self.limits.max_replay_values
        {
            return Err("generated preflight logger changed its transcript buffers".to_string());
        }
        if ffi.memory_dirty_pages != dirty_pages.memory.as_ptr().cast_mut()
            || ffi.public_values_dirty_pages != dirty_pages.public_values.as_ptr().cast_mut()
            || ffi.memory_dirty_page_words != dirty_pages.memory.len() as u64
            || ffi.public_values_dirty_page_words != dirty_pages.public_values.len() as u64
        {
            return Err("generated preflight logger changed its dirty-page buffers".to_string());
        }
        if ffi.checkpoint_interval != self.limits.checkpoint_interval
            || ffi.instruction_limit != self.limits.max_instructions
            || ffi.padding != 0
        {
            return Err("generated preflight logger changed its execution limits".to_string());
        }
        Ok(())
    }
}

fn validate_checkpoints(
    checkpoints: &[RvrCheckpoint],
    final_timestamp: u32,
    final_retired: u32,
    replay_value_len: usize,
    checkpoint_interval: u32,
    last_checkpoint_retired: u32,
) -> Result<(), String> {
    let mut previous_timestamp = 1;
    let mut previous_retired = 0;
    let mut previous_replay_value_cursor = 0;
    for checkpoint in checkpoints {
        let replay_value_cursor = usize::try_from(checkpoint.replay_value_cursor)
            .map_err(|_| "preflight replay-value cursor exceeds usize".to_string())?;
        if checkpoint.timestamp < previous_timestamp || checkpoint.timestamp > final_timestamp {
            return Err("preflight timestamps are not monotonic".to_string());
        }
        if checkpoint.retired < previous_retired || checkpoint.retired > final_retired {
            return Err("preflight checkpoint instruction counts are not monotonic".to_string());
        }
        if checkpoint.retired - previous_retired < checkpoint_interval {
            return Err("preflight checkpoint interval was not respected".to_string());
        }
        if replay_value_cursor < previous_replay_value_cursor
            || replay_value_cursor > replay_value_len
        {
            return Err("preflight replay-value cursors are not monotonic".to_string());
        }
        previous_timestamp = checkpoint.timestamp;
        previous_retired = checkpoint.retired;
        previous_replay_value_cursor = replay_value_cursor;
    }
    if last_checkpoint_retired != previous_retired {
        return Err("preflight last checkpoint instruction count is inconsistent".to_string());
    }
    Ok(())
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

    /// Persist the compiled shared library into `dir`.
    ///
    /// Loading requires the same VM executable and execution configuration.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        let suffix = self.inner.compiled.execution_kind().artifact_suffix();
        let dest_lib = self.inner.compiled.lib_file_name_with_suffix(suffix)?;
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
        self.execute_from_state_inner(state, limits, false, None)
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
        self.execute_from_state_inner(state, limits, true, None)
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
            self.execute_from_state_inner(state, limits, true, None)?,
            segment,
        )
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
            self.execute_from_state_inner(state, limits, true, Some(reuse))?,
            segment,
        )
    }

    fn execute_from_state_inner(
        &self,
        mut state: VmState<GuestMemory>,
        limits: PreflightLimits,
        allow_suspended: bool,
        reuse: Option<PreflightTranscript>,
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
                        limits,
                        self.inner.system_config.memory_config.timestamp_max_bits,
                        allow_suspended,
                        reuse,
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
mod tests {
    use super::*;

    #[test]
    fn derives_a_safe_checkpoint_capacity() {
        let limits = PreflightLimits::new(1025, 0, 512).validated().unwrap();
        assert_eq!(limits.max_checkpoints, 3);
    }

    #[test]
    fn rejects_zero_checkpoint_interval() {
        let error = PreflightLimits::new(1, 0, 0).validated().unwrap_err();
        assert!(error.contains("must be nonzero"));
    }

    #[test]
    fn reused_buffers_grow_to_larger_limits() {
        let initial = PreflightBuffers::new(PreflightLimits::new(128, 8, 64)).unwrap();
        let transcript = PreflightTranscript {
            checkpoints: initial.checkpoints,
            replay_values: initial.replay_values,
        };
        let reused =
            PreflightBuffers::reuse(PreflightLimits::new(1024, 64, 64), transcript).unwrap();
        assert!(reused.checkpoints.capacity() >= 16);
        assert!(reused.replay_values.capacity() >= 64);
    }

    #[test]
    fn finalization_rejects_reported_values_beyond_limits() {
        let mut retired = PreflightBuffers::new(PreflightLimits::new(8, 4, 4)).unwrap();
        let mut retired_dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
        let mut retired_ffi = retired.ffi_state(&mut retired_dirty);
        retired_ffi.retired = 9;
        let error = unsafe { retired.finish(&retired_ffi, 29, &retired_dirty) }.unwrap_err();
        assert!(error.contains("beyond its 8 instruction limit"));

        let mut buffers = PreflightBuffers::new(PreflightLimits::new(8, 4, 4)).unwrap();
        let mut dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
        let mut ffi = buffers.ffi_state(&mut dirty);
        ffi.replay_value_log_len = 5;
        let error = unsafe { buffers.finish(&ffi, 29, &dirty) }.unwrap_err();
        assert!(error.contains("out-of-bounds length"));
    }

    #[test]
    fn finalization_enforces_timestamp_domain() {
        let mut buffers = PreflightBuffers::new(PreflightLimits::new(8, 0, 4)).unwrap();
        let mut dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
        let mut ffi = buffers.ffi_state(&mut dirty);
        ffi.timestamp = 4;
        let error = unsafe { buffers.finish(&ffi, 2, &dirty) }.unwrap_err();
        assert!(error.contains("outside the configured 2-bit domain"));
    }

    #[test]
    fn segment_limits_use_metered_counts() {
        let segment = Segment::new(9, 17, 5, vec![]);
        assert_eq!(
            limits_for_segment(&segment).unwrap(),
            PreflightLimits::new(17, 5, DEFAULT_CHECKPOINT_INTERVAL)
        );
    }

    #[test]
    fn segment_boundary_rejects_replay_value_count_mismatch() {
        let config = SystemConfig::default();
        let state = VmState::initial(&config, &Default::default(), 0, Streams::default());
        let execution = PreflightExecution {
            state,
            transcript: PreflightTranscript {
                checkpoints: vec![],
                replay_values: vec![1],
            },
            endpoint: PreflightEndpoint::Suspended,
            from_state: ExecutionState::new(0u32, 1u32),
            to_state: ExecutionState::new(0u32, 1u32),
            retired: 2,
        };
        let segment = Segment::new(0, 2, 2, vec![]);
        match require_segment_boundary(execution, &segment) {
            Err(ExecutionError::PreflightReplayValueCountMismatch { expected, actual }) => {
                assert_eq!((expected, actual), (2, 1));
            }
            _ => panic!("unexpected segment validation result"),
        }
    }

    #[test]
    fn merges_executor_only_deferral_dirty_pages_into_initial_image_metadata() {
        let mut memory = AddressMap::default();
        assert!(memory.touched_pages[DEFERRAL_AS as usize]
            .touched_byte_ranges(PAGE_SIZE)
            .is_empty());

        let mut dirty = PreflightDirtyPages::new(&memory).unwrap();
        dirty.deferral[0] = 1;
        dirty.merge_into(&mut memory);

        assert_eq!(
            memory.touched_pages[DEFERRAL_AS as usize].touched_byte_ranges(PAGE_SIZE),
            vec![(0, PAGE_SIZE)]
        );
    }
}
