//! Append-only RVR preflight execution.

use std::path::Path;

use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use rustc_hash::FxHashSet;
use rvr_openvm_lift::RvrRuntimeExtension;
use rvr_state::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PreflightState,
};

use super::{
    bridge::map_rvr_execute_error, compile::CompileError, execute::execute_preflight, RvrCompiled,
    RvrInitialImage,
};
use crate::{
    arch::{
        AddressSpaceHostLayout, ExecutionError, Streams, SystemConfig, VmState, BLOCK_FE_WIDTH,
    },
    system::memory::online::GuestMemory,
};

const _: () = assert!(BLOCK_FE_WIDTH == 4);

/// Limits for one preflight execution call.
///
/// The final program sentinel needs one additional program-log slot, which is
/// reserved by the executor automatically. Bounded execution suspends before a
/// whole basic block that would exceed `max_instructions`;
/// `max_memory_events` remains a hard error boundary and never suspends midway
/// through a block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RvrPreflightLimits {
    pub max_instructions: usize,
    pub max_memory_events: usize,
}

impl RvrPreflightLimits {
    pub const fn new(max_instructions: usize, max_memory_events: usize) -> Self {
        Self {
            max_instructions,
            max_memory_events,
        }
    }
}

/// Minimal append-only output from one RVR preflight run.
#[derive(Debug)]
pub struct RvrPreflightTranscript {
    pub program_log: Vec<PreflightProgramEvent>,
    pub memory_log: Vec<PreflightMemoryEvent>,
    pub initial_write_log: Vec<PreflightInitialWrite>,
}

/// Why a complete preflight transcript stopped. This stays beside the
/// transcript rather than adding another hot-path log field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RvrPreflightEndpoint {
    Terminated,
    Suspended {
        resume_pc: u32,
        final_timestamp: u32,
    },
}

/// State and transcript returned after successful termination or bounded suspension.
pub struct RvrPreflightExecution {
    pub state: VmState<GuestMemory>,
    pub transcript: RvrPreflightTranscript,
    pub endpoint: RvrPreflightEndpoint,
}

pub(crate) struct PreflightBuffers {
    program_log: Vec<PreflightProgramEvent>,
    program_limit: usize,
    memory_log: Vec<PreflightMemoryEvent>,
    memory_limit: usize,
    initial_write_candidates: Vec<PreflightInitialWrite>,
}

impl PreflightBuffers {
    pub(crate) fn new(limits: RvrPreflightLimits) -> Result<Self, String> {
        let program_capacity = limits
            .max_instructions
            .checked_add(1)
            .ok_or_else(|| "preflight program capacity overflow".to_string())?;
        let mut program_log = Vec::new();
        program_log
            .try_reserve_exact(program_capacity)
            .map_err(|error| format!("failed to reserve preflight program log: {error}"))?;
        let mut memory_log = Vec::new();
        memory_log
            .try_reserve_exact(limits.max_memory_events)
            .map_err(|error| format!("failed to reserve preflight memory log: {error}"))?;
        let mut initial_write_candidates = Vec::new();
        initial_write_candidates
            .try_reserve_exact(limits.max_memory_events)
            .map_err(|error| format!("failed to reserve preflight initial-write log: {error}"))?;
        Ok(Self {
            program_log,
            program_limit: limits.max_instructions,
            memory_log,
            memory_limit: limits.max_memory_events,
            // The generated hot path appends one candidate per write. Cold
            // finalization retains only first-event writes.
            initial_write_candidates,
        })
    }

    pub(crate) fn ffi_state(&mut self) -> PreflightState {
        PreflightState {
            program_log: self.program_log.as_mut_ptr(),
            memory_log: self.memory_log.as_mut_ptr(),
            initial_write_log: self.initial_write_candidates.as_mut_ptr(),
            program_log_len: 0,
            program_log_cap: self.program_limit as u64,
            memory_log_len: 0,
            memory_log_cap: self.memory_limit as u64,
            initial_write_log_len: 0,
            initial_write_log_cap: self.memory_limit as u64,
            timestamp: 1,
            error: 0,
        }
    }

    /// # Safety
    ///
    /// `ffi` must be the state returned by [`Self::ffi_state`] and no pointer
    /// or capacity may have changed during generated execution.
    pub(crate) unsafe fn finish(
        mut self,
        ffi: &PreflightState,
        final_pc: u32,
        timestamp_max_bits: usize,
    ) -> Result<RvrPreflightTranscript, String> {
        if ffi.error != 0 {
            return Err(format!(
                "generated preflight logger failed with code {}",
                ffi.error
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
        let program_len = usize::try_from(ffi.program_log_len)
            .map_err(|_| "preflight program length does not fit usize".to_string())?;
        let memory_len = usize::try_from(ffi.memory_log_len)
            .map_err(|_| "preflight memory length does not fit usize".to_string())?;
        let initial_len = usize::try_from(ffi.initial_write_log_len)
            .map_err(|_| "preflight initial-write length does not fit usize".to_string())?;
        if program_len > self.program_limit
            || memory_len > self.memory_limit
            || initial_len > self.memory_limit
            || program_len > self.program_log.capacity()
            || memory_len > self.memory_log.capacity()
            || initial_len > self.initial_write_candidates.capacity()
        {
            return Err("generated preflight logger returned an out-of-bounds length".to_string());
        }

        // SAFETY: generated C initialized exactly these prefixes after checking
        // the capacities above.
        unsafe {
            self.program_log.set_len(program_len);
            self.memory_log.set_len(memory_len);
            self.initial_write_candidates.set_len(initial_len);
        }
        if self.program_log.len() == self.program_log.capacity() {
            return Err("preflight program log has no capacity for its final sentinel".to_string());
        }
        self.program_log.push(PreflightProgramEvent {
            pc: final_pc,
            timestamp: ffi.timestamp,
        });

        let mut seen = FxHashSet::default();
        let mut candidate_index = 0usize;
        let mut initial_write_log = Vec::new();
        for event in &self.memory_log {
            let address_space = event.address_space();
            let is_write = event.is_write();
            let key = ((address_space as u64) << 32) | event.pointer as u64;
            let first_event = seen.insert(key);
            if is_write {
                let candidate = *self
                    .initial_write_candidates
                    .get(candidate_index)
                    .ok_or_else(|| "missing initial-write candidate".to_string())?;
                candidate_index += 1;
                if candidate.address_space != address_space || candidate.pointer != event.pointer {
                    return Err("initial-write candidate is out of order".to_string());
                }
                if first_event {
                    initial_write_log.push(candidate);
                }
            }
        }
        if candidate_index != self.initial_write_candidates.len() {
            return Err("unused initial-write candidates remain".to_string());
        }

        Ok(RvrPreflightTranscript {
            program_log: self.program_log,
            memory_log: self.memory_log,
            initial_write_log,
        })
    }
}

pub(crate) fn extend_touched_pages(
    state: &mut VmState<GuestMemory>,
    transcript: &RvrPreflightTranscript,
) -> Result<(), String> {
    for event in &transcript.memory_log {
        if !event.is_write() {
            continue;
        }
        let address_space = event.address_space();
        if address_space != RV64_REGISTER_AS && address_space != RV64_MEMORY_AS {
            return Err(format!(
                "unsupported preflight address space {address_space} in first executor milestone"
            ));
        }
        let address_space_index = address_space as usize;
        let cell_bytes = state.memory.memory.config[address_space_index]
            .layout
            .size();
        let byte_start = (event.pointer as usize)
            .checked_mul(cell_bytes)
            .ok_or_else(|| "preflight touched-page pointer overflow".to_string())?;
        state.memory.memory.touched_pages[address_space_index]
            .mark_byte_range(byte_start, BLOCK_FE_WIDTH * cell_bytes);
    }
    Ok(())
}

struct RvrPreflightInstanceInner<'a> {
    system_config: &'a SystemConfig,
    initial_image: RvrInitialImage,
    compiled: RvrCompiled,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

/// Compiled append-only RVR preflight executor.
pub struct RvrPreflightInstance<'a> {
    inner: RvrPreflightInstanceInner<'a>,
}

static_assertions::assert_impl_all!(RvrPreflightInstance<'static>: Send, Sync);

impl<'a> RvrPreflightInstance<'a> {
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            inner: RvrPreflightInstanceInner {
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
        limits: RvrPreflightLimits,
    ) -> Result<RvrPreflightExecution, ExecutionError> {
        self.execute_from_state(self.create_initial_vm_state(inputs), limits)
    }

    /// Executes until successful termination. A generated suspension is treated
    /// as an error.
    ///
    /// Consumes `state` as the transaction boundary. On any execution or
    /// finalization error, the mutated working state is discarded and neither
    /// state nor transcript is returned.
    pub fn execute_from_state(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrPreflightLimits,
    ) -> Result<RvrPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, false)
    }

    /// Executes until either successful termination or the next whole basic
    /// block would exceed `limits.max_instructions`. Memory-log exhaustion
    /// remains an error.
    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        limits: RvrPreflightLimits,
    ) -> Result<RvrPreflightExecution, ExecutionError> {
        self.execute_from_state_for(self.create_initial_vm_state(inputs), limits)
    }

    /// Resumes from `state` and returns it only after successful termination or
    /// suspension at a whole-basic-block boundary. Any error discards the
    /// consumed working state.
    pub fn execute_from_state_for(
        &self,
        state: VmState<GuestMemory>,
        limits: RvrPreflightLimits,
    ) -> Result<RvrPreflightExecution, ExecutionError> {
        self.execute_from_state_inner(state, limits, true)
    }

    fn execute_from_state_inner(
        &self,
        mut state: VmState<GuestMemory>,
        limits: RvrPreflightLimits,
        allow_suspended: bool,
    ) -> Result<RvrPreflightExecution, ExecutionError> {
        let (transcript, endpoint) = execute_preflight(
            &self.inner.compiled,
            &self.inner.runtime_hooks,
            &mut state,
            limits,
            self.inner.system_config.memory_config.timestamp_max_bits,
            allow_suspended,
        )
        .map_err(map_rvr_execute_error)?;
        extend_touched_pages(&mut state, &transcript).map_err(ExecutionError::RvrExecution)?;
        Ok(RvrPreflightExecution {
            state,
            transcript,
            endpoint,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalization_enforces_the_exact_timestamp_domain_boundary() {
        let mut accepted = PreflightBuffers::new(RvrPreflightLimits::new(0, 0)).unwrap();
        let mut accepted_ffi = accepted.ffi_state();
        accepted_ffi.timestamp = 3;
        // SAFETY: `accepted_ffi` was created by `accepted`, and no pointers or
        // capacities were changed.
        unsafe { accepted.finish(&accepted_ffi, 0, 2) }.unwrap();

        let mut rejected = PreflightBuffers::new(RvrPreflightLimits::new(0, 0)).unwrap();
        let mut rejected_ffi = rejected.ffi_state();
        rejected_ffi.timestamp = 4;
        // SAFETY: `rejected_ffi` was created by `rejected`, and no pointers or
        // capacities were changed.
        let error = unsafe { rejected.finish(&rejected_ffi, 0, 2) }.unwrap_err();
        assert!(error.contains("outside the configured 2-bit domain"));
    }
}
