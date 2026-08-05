use openvm_instructions::{
    riscv::{MEMORY_AS, REGISTER_AS},
    DEFERRAL_AS,
};
use rvr_state::{PreflightTranscriptState, RvrCheckpoint, PREFLIGHT_DIRTY_PAGE_BYTES};

use super::{PreflightLimits, PreflightTranscript, ValidatedLimits};
use crate::system::memory::{
    online::{LinearMemory, PAGE_SIZE},
    AddressMap,
};

const _: () = assert!(PREFLIGHT_DIRTY_PAGE_BYTES == PAGE_SIZE);

pub(crate) struct PreflightBuffers {
    pub(super) checkpoints: Vec<RvrCheckpoint>,
    pub(super) replay_values: Vec<u64>,
    limits: ValidatedLimits,
}

/// Executor-only sparse-upload metadata. This is intentionally absent from
/// [`PreflightTranscript`]: replay does not consume it.
pub(crate) struct PreflightDirtyPages {
    memory: Box<[u64]>,
    pub(super) deferral: Box<[u64]>,
}

impl PreflightDirtyPages {
    pub(crate) fn new(memory: &AddressMap) -> Result<Self, String> {
        Ok(Self {
            memory: zeroed_dirty_page_words(memory.mem[MEMORY_AS as usize].size())?,
            deferral: zeroed_dirty_page_words(memory.mem[DEFERRAL_AS as usize].size())?,
        })
    }

    pub(crate) fn deferral_mut(&mut self) -> &mut [u64] {
        &mut self.deferral
    }

    pub(crate) fn merge_into(&self, memory: &mut AddressMap) {
        merge_dirty_page_words(memory, MEMORY_AS, &self.memory);
        merge_dirty_page_words(memory, DEFERRAL_AS, &self.deferral);

        // Generated execution keeps registers in RvState and copies them back
        // only after a successful execution boundary. Mark their single page
        // when preflight finalizes successfully.
        if memory.mem[REGISTER_AS as usize].size() != 0 {
            memory.touched_pages[REGISTER_AS as usize].mark_byte_range(0, 1);
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
            memory_dirty_page_words: dirty_pages.memory.len() as u64,
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
            || ffi.memory_dirty_page_words != dirty_pages.memory.len() as u64
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
