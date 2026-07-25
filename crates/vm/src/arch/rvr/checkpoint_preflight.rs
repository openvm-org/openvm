//! Compact checkpoint-and-residual RVR preflight execution.
//!
//! Serial execution still uses mutable random-access VM memory. Its
//! authoritative preflight output contains only periodic architectural
//! checkpoints and ordered residual values that deterministic replay cannot
//! recover from the program and segment-start state. GPU expansion converts
//! those arrays into a read-only logical execution history for parallel
//! tracegen.
//!
//! Dirty-page bitsets are transfer metadata, not transcript data: they identify
//! host writes that must be copied before the next segment. Proof-visible reads
//! and their predecessor timestamps are reconstructed later by GPU chronology.

use std::path::Path;

use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use rvr_openvm_lift::RvrRuntimeExtension;
use rvr_state::{CheckpointPreflightState, RvrCheckpoint, CHECKPOINT_DIRTY_PAGE_BYTES};

use super::{
    bridge::map_rvr_execute_error, compile::CompileError, execute::execute_checkpoint_preflight,
    RvrCompiled, RvrInitialImage, RvrPreflightEndpoint,
};
use crate::{
    arch::{ExecutionError, ExecutionState, Streams, SystemConfig, VmState},
    system::memory::{
        online::{GuestMemory, LinearMemory, PAGE_SIZE},
        AddressMap,
    },
};

const _: () = assert!(CHECKPOINT_DIRTY_PAGE_BYTES == PAGE_SIZE);

/// Resource limits for one checkpoint-preflight execution call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RvrCheckpointPreflightLimits {
    pub max_instructions: usize,
    pub max_residuals: usize,
    pub checkpoint_interval: usize,
}

impl RvrCheckpointPreflightLimits {
    pub const fn new(
        max_instructions: usize,
        max_residuals: usize,
        checkpoint_interval: usize,
    ) -> Self {
        Self {
            max_instructions,
            max_residuals,
            checkpoint_interval,
        }
    }

    fn validated(self) -> Result<ValidatedLimits, String> {
        let max_instructions = u32::try_from(self.max_instructions)
            .map_err(|_| "checkpoint-preflight instruction limit exceeds u32".to_string())?;
        let max_residuals_u32 = u32::try_from(self.max_residuals).map_err(|_| {
            "checkpoint-preflight residual limit exceeds the u32 checkpoint-cursor domain"
                .to_string()
        })?;
        let max_residuals = u64::from(max_residuals_u32);
        let checkpoint_interval = u32::try_from(self.checkpoint_interval)
            .map_err(|_| "checkpoint-preflight interval exceeds u32".to_string())?;
        if checkpoint_interval == 0 {
            return Err("checkpoint-preflight interval must be nonzero".to_string());
        }

        // Every interior checkpoint advances the last-checkpoint ordinal by
        // at least one interval. Ceiling division is a simple safe bound even
        // when basic blocks overshoot an interval boundary.
        let max_checkpoints = (self.max_instructions / self.checkpoint_interval)
            .checked_add(usize::from(
                !self
                    .max_instructions
                    .is_multiple_of(self.checkpoint_interval),
            ))
            .ok_or_else(|| "checkpoint-preflight checkpoint bound overflow".to_string())?;
        let max_checkpoints_u64 = u64::try_from(max_checkpoints)
            .map_err(|_| "checkpoint-preflight checkpoint limit exceeds u64".to_string())?;

        Ok(ValidatedLimits {
            max_instructions,
            max_residuals,
            checkpoint_interval,
            max_checkpoints,
            max_checkpoints_u64,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedLimits {
    max_instructions: u32,
    max_residuals: u64,
    checkpoint_interval: u32,
    max_checkpoints: usize,
    max_checkpoints_u64: u64,
}

/// Minimal authoritative output of checkpoint preflight.
#[derive(Debug, Default)]
pub struct RvrCheckpointPreflightTranscript {
    pub checkpoints: Vec<RvrCheckpoint>,
    pub residuals: Vec<u64>,
}

/// State and compact transcript returned by experimental checkpoint preflight.
pub struct RvrCheckpointPreflightExecution {
    pub state: VmState<GuestMemory>,
    pub transcript: RvrCheckpointPreflightTranscript,
    pub endpoint: RvrPreflightEndpoint,
    /// Initial execution-bus boundary. The initial register and memory image
    /// are supplied separately by the caller; they are not duplicated here.
    pub from_state: ExecutionState<u32>,
    /// Final execution-bus boundary. The final registers and memory live in
    /// `state`; they are not copied into the transcript.
    pub to_state: ExecutionState<u32>,
    pub retired: u32,
}

pub(crate) struct CheckpointPreflightBuffers {
    checkpoints: Vec<RvrCheckpoint>,
    residuals: Vec<u64>,
    limits: ValidatedLimits,
}

/// Executor-only sparse-upload metadata. This is intentionally absent from
/// [`RvrCheckpointPreflightTranscript`]: replay does not consume it.
pub(crate) struct CheckpointDirtyPages {
    memory: Box<[u64]>,
    public_values: Box<[u64]>,
    deferral: Box<[u64]>,
}

impl CheckpointDirtyPages {
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
        // at the same successful checkpoint finalization boundary.
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
        .map_err(|error| format!("failed to reserve checkpoint dirty-page bits: {error}"))?;
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

impl CheckpointPreflightBuffers {
    pub(crate) fn new(limits: RvrCheckpointPreflightLimits) -> Result<Self, String> {
        Self::with_transcript(limits, RvrCheckpointPreflightTranscript::default())
    }

    pub(crate) fn reuse(
        limits: RvrCheckpointPreflightLimits,
        transcript: RvrCheckpointPreflightTranscript,
    ) -> Result<Self, String> {
        Self::with_transcript(limits, transcript)
    }

    fn with_transcript(
        limits: RvrCheckpointPreflightLimits,
        mut transcript: RvrCheckpointPreflightTranscript,
    ) -> Result<Self, String> {
        let limits = limits.validated()?;
        transcript.checkpoints.clear();
        transcript.residuals.clear();
        if transcript.checkpoints.capacity() < limits.max_checkpoints {
            transcript
                .checkpoints
                .try_reserve_exact(limits.max_checkpoints)
                .map_err(|error| {
                    format!("failed to reserve checkpoint-preflight checkpoints: {error}")
                })?;
        }
        let max_residuals = usize::try_from(limits.max_residuals)
            .map_err(|_| "checkpoint-preflight residual limit exceeds usize".to_string())?;
        if transcript.residuals.capacity() < max_residuals {
            transcript
                .residuals
                .try_reserve_exact(max_residuals)
                .map_err(|error| {
                    format!("failed to reserve checkpoint-preflight residuals: {error}")
                })?;
        }
        Ok(Self {
            checkpoints: transcript.checkpoints,
            residuals: transcript.residuals,
            limits,
        })
    }

    pub(crate) fn ffi_state(
        &mut self,
        dirty_pages: &mut CheckpointDirtyPages,
    ) -> CheckpointPreflightState {
        CheckpointPreflightState {
            checkpoint_log: self.checkpoints.as_mut_ptr(),
            residual_log: self.residuals.as_mut_ptr(),
            checkpoint_log_len: 0,
            checkpoint_log_cap: self.limits.max_checkpoints_u64,
            residual_log_len: 0,
            residual_log_cap: self.limits.max_residuals,
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
        ffi: &CheckpointPreflightState,
        timestamp_max_bits: usize,
        dirty_pages: &CheckpointDirtyPages,
    ) -> Result<(RvrCheckpointPreflightTranscript, u32, u32), String> {
        if ffi.error != 0 {
            return Err(format!(
                "generated checkpoint-preflight logger failed with code {}",
                ffi.error
            ));
        }
        if ffi.checkpoint_log != self.checkpoints.as_mut_ptr()
            || ffi.residual_log != self.residuals.as_mut_ptr()
            || ffi.checkpoint_log_cap != self.limits.max_checkpoints_u64
            || ffi.residual_log_cap != self.limits.max_residuals
            || ffi.checkpoint_interval != self.limits.checkpoint_interval
            || ffi.instruction_limit != self.limits.max_instructions
            || ffi.memory_dirty_pages != dirty_pages.memory.as_ptr().cast_mut()
            || ffi.public_values_dirty_pages != dirty_pages.public_values.as_ptr().cast_mut()
            || ffi.memory_dirty_page_words != dirty_pages.memory.len() as u64
            || ffi.public_values_dirty_page_words != dirty_pages.public_values.len() as u64
            || ffi.padding != 0
        {
            return Err("generated checkpoint-preflight logger changed its input ABI".to_string());
        }

        let checkpoint_len = usize::try_from(ffi.checkpoint_log_len)
            .map_err(|_| "checkpoint-preflight checkpoint length exceeds usize".to_string())?;
        let residual_len = usize::try_from(ffi.residual_log_len)
            .map_err(|_| "checkpoint-preflight residual length exceeds usize".to_string())?;
        if checkpoint_len > self.limits.max_checkpoints
            || checkpoint_len > self.checkpoints.capacity()
            || residual_len > self.residuals.capacity()
            || ffi.residual_log_len > self.limits.max_residuals
        {
            return Err(
                "generated checkpoint-preflight logger returned an out-of-bounds length"
                    .to_string(),
            );
        }
        if ffi.retired > self.limits.max_instructions {
            return Err(format!(
                "checkpoint-preflight retired {} instructions beyond its {} instruction limit",
                ffi.retired, self.limits.max_instructions
            ));
        }

        let timestamp_limit =
            1u32.checked_shl(u32::try_from(timestamp_max_bits).map_err(|_| {
                "checkpoint-preflight timestamp width does not fit u32".to_string()
            })?)
            .ok_or_else(|| {
                "checkpoint-preflight timestamp width must be less than 32".to_string()
            })?;
        if ffi.timestamp >= timestamp_limit {
            return Err(format!(
                "checkpoint-preflight final timestamp {} is outside the configured {timestamp_max_bits}-bit domain",
                ffi.timestamp
            ));
        }

        // SAFETY: the generated logger initialized exactly these prefixes and
        // the returned lengths were checked against the original capacities.
        unsafe {
            self.checkpoints.set_len(checkpoint_len);
            self.residuals.set_len(residual_len);
        }
        validate_checkpoints(
            &self.checkpoints,
            ffi.timestamp,
            ffi.retired,
            residual_len,
            self.limits.checkpoint_interval,
            ffi.last_checkpoint_retired,
        )?;

        Ok((
            RvrCheckpointPreflightTranscript {
                checkpoints: self.checkpoints,
                residuals: self.residuals,
            },
            ffi.timestamp,
            ffi.retired,
        ))
    }
}

fn validate_checkpoints(
    checkpoints: &[RvrCheckpoint],
    final_timestamp: u32,
    final_retired: u32,
    residual_len: usize,
    checkpoint_interval: u32,
    last_checkpoint_retired: u32,
) -> Result<(), String> {
    let mut previous_timestamp = 1;
    let mut previous_retired = 0;
    let mut previous_residual = 0;
    for checkpoint in checkpoints {
        let residual_cursor = usize::try_from(checkpoint.residual_cursor)
            .map_err(|_| "checkpoint-preflight residual cursor exceeds usize".to_string())?;
        if checkpoint.timestamp < previous_timestamp || checkpoint.timestamp > final_timestamp {
            return Err("checkpoint-preflight timestamps are not monotonic".to_string());
        }
        if checkpoint.retired < previous_retired || checkpoint.retired > final_retired {
            return Err("checkpoint-preflight retired ordinals are not monotonic".to_string());
        }
        if checkpoint.retired - previous_retired < checkpoint_interval {
            return Err("checkpoint-preflight checkpoint interval was not respected".to_string());
        }
        if residual_cursor < previous_residual || residual_cursor > residual_len {
            return Err("checkpoint-preflight residual cursors are not monotonic".to_string());
        }
        previous_timestamp = checkpoint.timestamp;
        previous_retired = checkpoint.retired;
        previous_residual = residual_cursor;
    }
    if last_checkpoint_retired != previous_retired {
        return Err("checkpoint-preflight last-checkpoint cursor is inconsistent".to_string());
    }
    Ok(())
}

struct RvrCheckpointPreflightInstanceInner<'a> {
    system_config: &'a SystemConfig,
    initial_image: RvrInitialImage,
    compiled: RvrCompiled,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

/// Compiled experimental checkpoint-preflight executor.
pub struct RvrCheckpointPreflightInstance<'a> {
    inner: RvrCheckpointPreflightInstanceInner<'a>,
}

static_assertions::assert_impl_all!(RvrCheckpointPreflightInstance<'static>: Send, Sync);

impl<'a> RvrCheckpointPreflightInstance<'a> {
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            inner: RvrCheckpointPreflightInstanceInner {
                system_config,
                initial_image,
                compiled,
                runtime_hooks,
            },
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner
            .initial_image
            .create_vm_state(self.inner.system_config, inputs)
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
        limits: RvrCheckpointPreflightLimits,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state(self.create_initial_vm_state(inputs), limits)
    }

    pub fn execute_from_state(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, false, None)
    }

    pub fn execute_from_state_reusing(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
        reuse: RvrCheckpointPreflightTranscript,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, false, Some(reuse))
    }

    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        limits: RvrCheckpointPreflightLimits,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state_for(self.create_initial_vm_state(inputs), limits)
    }

    pub fn execute_from_state_for(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, true, None)
    }

    /// Execute exactly the instruction count in `limits` from `state`.
    ///
    /// Metered continuation boundaries use this entry point so an early
    /// termination or block-boundary mismatch is rejected before the returned
    /// state can be carried into the next segment.
    pub fn execute_from_state_for_exact(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        require_exact_retired(self.execute_from_state_for(state, limits)?, limits)
    }

    pub fn execute_from_state_for_reusing(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
        reuse: RvrCheckpointPreflightTranscript,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, true, Some(reuse))
    }

    /// Reusing variant of [`Self::execute_from_state_for_exact`].
    pub fn execute_from_state_for_exact_reusing(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
        reuse: RvrCheckpointPreflightTranscript,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        require_exact_retired(
            self.execute_from_state_for_reusing(state, limits, reuse)?,
            limits,
        )
    }

    fn execute_from_state_inner(
        &self,
        mut state: VmState<GuestMemory>,
        limits: RvrCheckpointPreflightLimits,
        allow_suspended: bool,
        reuse: Option<RvrCheckpointPreflightTranscript>,
    ) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
        let from_state = ExecutionState::new(state.pc(), 1u32);
        let (transcript, endpoint, final_timestamp, retired) =
            tracing::info_span!("execute_checkpoint_preflight")
                .in_scope(|| {
                    execute_checkpoint_preflight(
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
        let to_state = ExecutionState::new(state.pc(), final_timestamp);
        Ok(RvrCheckpointPreflightExecution {
            state,
            transcript,
            endpoint,
            from_state,
            to_state,
            retired,
        })
    }

    pub fn save(&self, dir: &Path) -> Result<std::path::PathBuf, CompileError> {
        let suffix = self.inner.compiled.execution_kind().artifact_suffix();
        let dest_lib = self.inner.compiled.lib_file_name_with_suffix(suffix)?;
        self.inner.compiled.save_artifact(&dir.join(dest_lib))
    }

    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.inner.compiled.save_generated_sources(dir)
    }
}

fn require_exact_retired(
    execution: RvrCheckpointPreflightExecution,
    limits: RvrCheckpointPreflightLimits,
) -> Result<RvrCheckpointPreflightExecution, ExecutionError> {
    let expected_retired = u32::try_from(limits.max_instructions).map_err(|_| {
        ExecutionError::RvrExecution(
            "checkpoint-preflight instruction limit exceeds u32".to_string(),
        )
    })?;
    if execution.retired != expected_retired {
        return Err(ExecutionError::RvrExecution(format!(
            "checkpoint execution retired {} instructions, expected {expected_retired}",
            execution.retired
        )));
    }
    Ok(execution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_a_safe_checkpoint_capacity() {
        let limits = RvrCheckpointPreflightLimits::new(1025, 0, 512)
            .validated()
            .unwrap();
        assert_eq!(limits.max_checkpoints, 3);
    }

    #[test]
    fn rejects_zero_checkpoint_interval() {
        let error = RvrCheckpointPreflightLimits::new(1, 0, 0)
            .validated()
            .unwrap_err();
        assert!(error.contains("must be nonzero"));
    }

    #[test]
    fn reused_buffers_grow_to_larger_limits() {
        let initial =
            CheckpointPreflightBuffers::new(RvrCheckpointPreflightLimits::new(128, 8, 64)).unwrap();
        let transcript = RvrCheckpointPreflightTranscript {
            checkpoints: initial.checkpoints,
            residuals: initial.residuals,
        };
        let reused = CheckpointPreflightBuffers::reuse(
            RvrCheckpointPreflightLimits::new(1024, 64, 64),
            transcript,
        )
        .unwrap();
        assert!(reused.checkpoints.capacity() >= 16);
        assert!(reused.residuals.capacity() >= 64);
    }

    #[test]
    fn finalization_rejects_reported_values_beyond_limits() {
        let mut retired =
            CheckpointPreflightBuffers::new(RvrCheckpointPreflightLimits::new(8, 4, 4)).unwrap();
        let mut retired_dirty = CheckpointDirtyPages::new(&AddressMap::default()).unwrap();
        let mut retired_ffi = retired.ffi_state(&mut retired_dirty);
        retired_ffi.retired = 9;
        let error = unsafe { retired.finish(&retired_ffi, 29, &retired_dirty) }.unwrap_err();
        assert!(error.contains("beyond its 8 instruction limit"));

        let mut residuals =
            CheckpointPreflightBuffers::new(RvrCheckpointPreflightLimits::new(8, 4, 4)).unwrap();
        let mut residuals_dirty = CheckpointDirtyPages::new(&AddressMap::default()).unwrap();
        let mut residuals_ffi = residuals.ffi_state(&mut residuals_dirty);
        residuals_ffi.residual_log_len = 5;
        let error = unsafe { residuals.finish(&residuals_ffi, 29, &residuals_dirty) }.unwrap_err();
        assert!(error.contains("out-of-bounds length"));
    }

    #[test]
    fn finalization_enforces_timestamp_domain() {
        let mut buffers =
            CheckpointPreflightBuffers::new(RvrCheckpointPreflightLimits::new(8, 0, 4)).unwrap();
        let mut dirty = CheckpointDirtyPages::new(&AddressMap::default()).unwrap();
        let mut ffi = buffers.ffi_state(&mut dirty);
        ffi.timestamp = 4;
        let error = unsafe { buffers.finish(&ffi, 2, &dirty) }.unwrap_err();
        assert!(error.contains("outside the configured 2-bit domain"));
    }

    #[test]
    fn merges_executor_only_deferral_dirty_pages_into_initial_image_metadata() {
        let mut memory = AddressMap::default();
        assert!(memory.touched_pages[DEFERRAL_AS as usize]
            .touched_byte_ranges(PAGE_SIZE)
            .is_empty());

        let mut dirty = CheckpointDirtyPages::new(&memory).unwrap();
        dirty.deferral[0] = 1;
        dirty.merge_into(&mut memory);

        assert_eq!(
            memory.touched_pages[DEFERRAL_AS as usize].touched_byte_ranges(PAGE_SIZE),
            vec![(0, PAGE_SIZE)]
        );
    }
}
