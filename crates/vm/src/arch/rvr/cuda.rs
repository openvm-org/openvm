//! GPU-owned copies of the immutable program and append-only preflight logs.

use std::sync::Arc;
#[cfg(feature = "test-utils")]
use std::time::{Duration, Instant};

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::{CudaError, MemCopyError},
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    instruction::Instruction, program::Program, LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent};
use thiserror::Error;

#[cfg(feature = "test-utils")]
use super::postflight::RvrReplayData;
use super::{postflight::RvrReplayStep, RvrPreflightEndpoint, RvrPreflightTranscript};
use crate::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    cuda_abi::rvr_postflight,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RvrReplayInstruction {
    /// Global opcode followed by the seven canonical instruction operands.
    pub words: [u32; 8],
}

const _: () = assert!(size_of::<RvrReplayInstruction>() == 32);

#[derive(Debug, Error)]
pub enum GpuRvrInputError {
    #[error("program opcode {0} does not fit the GPU replay ABI")]
    OpcodeTooLarge(usize),
    #[error("invalid RVR GPU memory configuration: {0}")]
    InvalidMemoryConfig(String),
    #[error("{0}")]
    InvalidTranscript(String),
    #[error("RVR replay input belongs to another CUDA device or stream")]
    ContextMismatch,
    #[error("RVR replay transcript belongs to another uploaded program")]
    ProgramMismatch,
    #[error("RVR replay plan belongs to another transcript segment")]
    SegmentMismatch,
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Copy(#[from] MemCopyError),
}

fn upload<T>(values: &[T], device_ctx: &GpuDeviceCtx) -> Result<DeviceBuffer<T>, MemCopyError> {
    if values.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        values.to_device_on(device_ctx)
    }
}

/// Static program data uploaded once and shared by every replayed segment.
pub struct GpuRvrProgram {
    instructions: DeviceBuffer<RvrReplayInstruction>,
    #[cfg(feature = "test-utils")]
    opcodes: Vec<u32>,
    active_opcodes: Vec<u32>,
    d_active_opcodes: DeviceBuffer<u32>,
    address_space_height: u32,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    pc_base: u32,
    device_ctx: GpuDeviceCtx,
    identity: Arc<()>,
}

impl GpuRvrProgram {
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuRvrInputError> {
        if memory_config.pointer_max_bits > u32::BITS as usize
            || memory_config.addr_space_height >= u32::BITS as usize
            || memory_config.timestamp_max_bits >= u32::BITS as usize
        {
            return Err(GpuRvrInputError::InvalidMemoryConfig(
                "address-space height, pointer width, and timestamp width must fit u32".to_string(),
            ));
        }
        let address_space_count = 1usize
            .checked_shl(memory_config.addr_space_height as u32)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        let expected_address_spaces = (ADDR_SPACE_OFFSET as usize)
            .checked_add(address_space_count)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        if memory_config.addr_spaces.len() != expected_address_spaces {
            return Err(GpuRvrInputError::InvalidMemoryConfig(format!(
                "expected {expected_address_spaces} address-space layouts, found {}",
                memory_config.addr_spaces.len()
            )));
        }
        let block_pointer_bits = memory_config
            .pointer_max_bits
            .checked_sub(BLOCK_FE_WIDTH.ilog2() as usize)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidMemoryConfig(
                    "pointer width is smaller than one memory block".to_string(),
                )
            })?;
        let label_bits = memory_config
            .addr_space_height
            .checked_add(block_pointer_bits)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidMemoryConfig(
                    "address-space and block-pointer label width overflow".to_string(),
                )
            })?;
        if label_bits > u32::BITS as usize {
            return Err(GpuRvrInputError::InvalidMemoryConfig(
                "address-space and block-pointer label does not fit u32".to_string(),
            ));
        }
        let instructions = program
            .instructions_and_debug_infos
            .iter()
            .map(|entry| match entry {
                Some((instruction, _)) => instruction_to_replay(instruction),
                None => Ok(RvrReplayInstruction {
                    words: [u32::MAX, 0, 0, 0, 0, 0, 0, 0],
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let opcodes: Vec<u32> = instructions
            .iter()
            .map(|instruction| instruction.words[0])
            .collect();
        let mut active_opcodes = opcodes
            .iter()
            .copied()
            .filter(|&opcode| opcode != u32::MAX)
            .collect::<Vec<_>>();
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            #[cfg(feature = "test-utils")]
            opcodes,
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            address_space_height: memory_config.addr_space_height as u32,
            pointer_max_bits: memory_config.pointer_max_bits as u32,
            timestamp_max_bits: memory_config.timestamp_max_bits as u32,
            pc_base: program.pc_base,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        })
    }

    pub fn instructions(&self) -> DeviceBufferView {
        self.instructions.view()
    }

    pub const fn pc_base(&self) -> u32 {
        self.pc_base
    }

    /// Upload one segment's logs and its derived replay work. Deriving the work
    /// through the uploaded program prevents mixing a plan with another static
    /// program that happens to have the same length.
    pub fn upload_transcript(
        &self,
        transcript: &RvrPreflightTranscript,
        endpoint: RvrPreflightEndpoint,
    ) -> Result<(GpuRvrTranscript, GpuRvrReplayPlan), GpuRvrInputError> {
        let segment_identity = Arc::new(());
        let gpu = GpuRvrTranscript::upload(
            transcript,
            self.address_space_height,
            self.pointer_max_bits,
            &self.device_ctx,
            self.identity.clone(),
            segment_identity.clone(),
        )?;
        let plan = GpuRvrReplayPlan::build(
            self,
            &gpu,
            endpoint,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((gpu, plan))
    }

    /// Benchmark-only split of synchronized raw-log upload plus memory indexing
    /// and synchronized GPU program indexing plus its small range-table D2H.
    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn upload_transcript_profiled(
        &self,
        transcript: &RvrPreflightTranscript,
        endpoint: RvrPreflightEndpoint,
    ) -> Result<(GpuRvrTranscript, GpuRvrReplayPlan, Duration, Duration), GpuRvrInputError> {
        self.device_ctx.stream.synchronize()?;
        let started = Instant::now();
        let segment_identity = Arc::new(());
        let transcript = GpuRvrTranscript::upload(
            transcript,
            self.address_space_height,
            self.pointer_max_bits,
            &self.device_ctx,
            self.identity.clone(),
            segment_identity.clone(),
        )?;
        self.device_ctx.stream.synchronize()?;
        let upload_time = started.elapsed();

        let started = Instant::now();
        let plan = GpuRvrReplayPlan::build(
            self,
            &transcript,
            endpoint,
            self.identity.clone(),
            segment_identity,
        )?;
        let index_time = started.elapsed();
        Ok((transcript, plan, index_time, upload_time))
    }

    pub fn ensure_replay_inputs(
        &self,
        transcript: &GpuRvrTranscript,
        plan: &GpuRvrReplayPlan,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), GpuRvrInputError> {
        ensure_same_context(&self.device_ctx, device_ctx)?;
        ensure_same_context(&transcript.device_ctx, device_ctx)?;
        ensure_same_context(&plan.device_ctx, device_ctx)?;
        if !Arc::ptr_eq(&self.identity, &transcript.program_identity)
            || !Arc::ptr_eq(&self.identity, &plan.program_identity)
        {
            return Err(GpuRvrInputError::ProgramMismatch);
        }
        if !Arc::ptr_eq(&transcript.segment_identity, &plan.segment_identity) {
            return Err(GpuRvrInputError::SegmentMismatch);
        }
        Ok(())
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn cpu_memory_predecessors(
        transcript: &RvrPreflightTranscript,
    ) -> Result<Vec<u32>, GpuRvrInputError> {
        super::postflight::build_memory_predecessors(
            &transcript.memory_log,
            &transcript.initial_write_log,
        )
        .map_err(|error| GpuRvrInputError::InvalidTranscript(error.to_string()))
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn cpu_replay_plan(
        &self,
        transcript: &RvrPreflightTranscript,
        endpoint: RvrPreflightEndpoint,
    ) -> Result<
        (
            Vec<[u32; 2]>,
            std::collections::BTreeMap<u32, std::ops::Range<usize>>,
        ),
        GpuRvrInputError,
    > {
        let replay = RvrReplayData::build(self.pc_base, &self.opcodes, transcript, endpoint)
            .map_err(|error| GpuRvrInputError::InvalidTranscript(error.to_string()))?;
        Ok((
            replay
                .steps()
                .iter()
                .map(|step| [step.program_index, step.memory_start])
                .collect(),
            replay.opcode_ranges().clone(),
        ))
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    /// Requested live-buffer payload, excluding allocator page rounding, the
    /// shared error word, raw logs, and the static program.
    pub fn gpu_index_memory_bytes(
        &self,
        transcript: &RvrPreflightTranscript,
    ) -> Result<(usize, usize), GpuRvrInputError> {
        let num_memory = transcript.memory_log.len();
        let num_entries = num_memory + transcript.initial_write_log.len();
        let num_steps = transcript.program_log.len().saturating_sub(1);
        let mut memory_temp = 0usize;
        let mut program_temp = 0usize;
        unsafe {
            rvr_postflight::memory_index_get_temp_bytes(
                num_entries,
                &mut memory_temp,
                self.device_ctx.stream.as_raw(),
            )?;
            rvr_postflight::program_index_get_temp_bytes(
                num_steps,
                &mut program_temp,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        // Both radix sorts use separate input and output buffers. Memory
        // embeds each source ordinal in its 64-bit sort key; program indexing
        // sorts an 8-byte step value beside each 32-bit opcode key.
        // They run sequentially, so peak incremental allocation is the larger
        // stage rather than their sum.
        let memory_sort_stage = 16 * num_entries + memory_temp;
        let memory_scatter_stage = 8 * num_entries + 4 * num_memory;
        let program_stage =
            4 * num_memory + 24 * num_steps + 8 * self.active_opcodes.len() + program_temp;
        let steady = 4 * num_memory + 8 * num_steps;
        Ok((
            memory_sort_stage
                .max(memory_scatter_stage)
                .max(program_stage),
            steady,
        ))
    }
}

fn ensure_same_context(
    expected: &GpuDeviceCtx,
    actual: &GpuDeviceCtx,
) -> Result<(), GpuRvrInputError> {
    if expected.device_id == actual.device_id && expected.stream == actual.stream {
        Ok(())
    } else {
        Err(GpuRvrInputError::ContextMismatch)
    }
}

fn instruction_to_replay<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Result<RvrReplayInstruction, GpuRvrInputError> {
    let opcode = u32::try_from(instruction.opcode.as_usize())
        .map_err(|_| GpuRvrInputError::OpcodeTooLarge(instruction.opcode.as_usize()))?;
    Ok(RvrReplayInstruction {
        words: [
            opcode,
            instruction.a.as_canonical_u32(),
            instruction.b.as_canonical_u32(),
            instruction.c.as_canonical_u32(),
            instruction.d.as_canonical_u32(),
            instruction.e.as_canonical_u32(),
            instruction.f.as_canonical_u32(),
            instruction.g.as_canonical_u32(),
        ],
    })
}

fn gpu_buffer<T>(len: usize, device_ctx: &GpuDeviceCtx) -> DeviceBuffer<T> {
    if len == 0 {
        DeviceBuffer::new()
    } else {
        DeviceBuffer::with_capacity_on(len, device_ctx)
    }
}

fn build_gpu_memory_index(
    memory: &DeviceBuffer<PreflightMemoryEvent>,
    seeds: &DeviceBuffer<PreflightInitialWrite>,
    address_space_height: u32,
    pointer_max_bits: u32,
    error: &DeviceBuffer<u32>,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<u32>, CudaError> {
    let num_entries = memory
        .len()
        .checked_add(seeds.len())
        .expect("memory index entry count overflow");
    let keys_in = gpu_buffer::<u64>(num_entries, device_ctx);
    let keys_out = gpu_buffer::<u64>(num_entries, device_ctx);

    let mut temp_bytes = 0usize;
    unsafe {
        rvr_postflight::memory_index_get_temp_bytes(
            num_entries,
            &mut temp_bytes,
            device_ctx.stream.as_raw(),
        )?;
    }
    let temp_storage = gpu_buffer::<u8>(temp_bytes, device_ctx);
    unsafe {
        rvr_postflight::memory_index_sort(
            memory.view(),
            seeds.view(),
            ADDR_SPACE_OFFSET,
            address_space_height,
            pointer_max_bits,
            &keys_in,
            &keys_out,
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    // Both frees are enqueued on this stream after the sort. Allocating the
    // output only afterwards keeps sort scratch and retained predecessors out
    // of the same peak without introducing a host synchronization.
    drop(keys_in);
    drop(temp_storage);
    let memory_predecessors = gpu_buffer::<u32>(memory.len(), device_ctx);
    unsafe {
        rvr_postflight::memory_index_scatter(
            memory.view(),
            seeds.len(),
            &keys_out,
            num_entries,
            &memory_predecessors,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    Ok(memory_predecessors)
}

/// Device-resident transcript and its one generic predecessor index.
///
/// This object is shared across all opcode kernels for the segment. `error` is
/// also shared so replay validation can fail closed without a per-chip copy.
/// Read it once after all replay kernels. A nonzero result is terminal for that
/// proving attempt because threads from other rows may already have updated
/// shared lookup histograms.
pub struct GpuRvrTranscript {
    program_log: DeviceBuffer<PreflightProgramEvent>,
    memory_log: DeviceBuffer<PreflightMemoryEvent>,
    initial_write_log: DeviceBuffer<PreflightInitialWrite>,
    memory_predecessors: DeviceBuffer<u32>,
    error: DeviceBuffer<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuRvrTranscript {
    fn upload(
        transcript: &RvrPreflightTranscript,
        address_space_height: u32,
        pointer_max_bits: u32,
        device_ctx: &GpuDeviceCtx,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuRvrInputError> {
        let packed_index_limit = (1usize << 31) - 1;
        if transcript.memory_log.len() >= packed_index_limit
            || transcript.initial_write_log.len() >= packed_index_limit
        {
            return Err(GpuRvrInputError::InvalidTranscript(
                "memory or initial-write log is too large for packed predecessor indexes"
                    .to_string(),
            ));
        }
        let program_log = upload(&transcript.program_log, device_ctx)?;
        let memory_log = upload(&transcript.memory_log, device_ctx)?;
        let initial_write_log = upload(&transcript.initial_write_log, device_ctx)?;
        let error = [0u32].to_device_on(device_ctx)?;
        let memory_predecessors = build_gpu_memory_index(
            &memory_log,
            &initial_write_log,
            address_space_height,
            pointer_max_bits,
            &error,
            device_ctx,
        )?;
        Ok(Self {
            program_log,
            memory_log,
            initial_write_log,
            memory_predecessors,
            error,
            device_ctx: device_ctx.clone(),
            program_identity,
            segment_identity,
        })
    }

    pub fn error_code(&self) -> Result<u32, MemCopyError> {
        Ok(self.error.to_host_on(&self.device_ctx)?[0])
    }

    pub fn program_log(&self) -> DeviceBufferView {
        self.program_log.view()
    }

    pub fn memory_log(&self) -> DeviceBufferView {
        self.memory_log.view()
    }

    pub fn initial_write_log(&self) -> DeviceBufferView {
        self.initial_write_log.view()
    }

    pub fn memory_predecessors(&self) -> DeviceBufferView {
        self.memory_predecessors.view()
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn memory_predecessors_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.memory_predecessors.to_host_on(&self.device_ctx)
    }

    pub fn error_ptr(&self) -> *mut u32 {
        self.error.as_mut_ptr()
    }
}

/// The opcode-partitioned replay work list, uploaded once per segment.
pub struct GpuRvrReplayPlan {
    steps: DeviceBuffer<RvrReplayStep>,
    opcode_ranges: std::collections::BTreeMap<u32, std::ops::Range<usize>>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuRvrReplayPlan {
    fn build(
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        endpoint: RvrPreflightEndpoint,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuRvrInputError> {
        let num_program_events = transcript.program_log.len();
        if num_program_events == 0 {
            return Err(GpuRvrInputError::InvalidTranscript(
                "transcript must contain a final sentinel".to_string(),
            ));
        }
        let num_steps = num_program_events - 1;
        if num_steps >= u32::MAX as usize {
            return Err(GpuRvrInputError::InvalidTranscript(
                "program log has more than u32::MAX entries".to_string(),
            ));
        }
        let opcode_keys_in = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let opcode_keys_out = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let steps_in = gpu_buffer::<RvrReplayStep>(num_steps, &program.device_ctx);
        let steps_out = gpu_buffer::<RvrReplayStep>(num_steps, &program.device_ctx);
        let ranges = gpu_buffer::<u32>(2 * program.active_opcodes.len(), &program.device_ctx);
        let mut temp_bytes = 0usize;
        unsafe {
            rvr_postflight::program_index_get_temp_bytes(
                num_steps,
                &mut temp_bytes,
                program.device_ctx.stream.as_raw(),
            )?;
        }
        let temp_storage = gpu_buffer::<u8>(temp_bytes, &program.device_ctx);
        let (endpoint_kind, resume_pc, final_timestamp) = match endpoint {
            RvrPreflightEndpoint::Terminated => (0, 0, 0),
            RvrPreflightEndpoint::Suspended {
                resume_pc,
                final_timestamp,
            } => (1, resume_pc, final_timestamp),
        };
        unsafe {
            rvr_postflight::program_index(
                program.instructions.view(),
                program.pc_base,
                transcript.program_log.view(),
                transcript.memory_log.view(),
                program.d_active_opcodes.view(),
                program.timestamp_max_bits,
                endpoint_kind,
                resume_pc,
                final_timestamp,
                SystemOpcode::TERMINATE.global_opcode().as_usize() as u32,
                &opcode_keys_in,
                &opcode_keys_out,
                steps_in.as_mut_raw_ptr(),
                steps_out.as_mut_raw_ptr(),
                &ranges,
                &temp_storage,
                temp_bytes,
                &transcript.error,
                program.device_ctx.stream.as_raw(),
            )?;
        }
        let ranges = ranges.to_host_on(&program.device_ctx)?;
        let error = transcript.error.to_host_on(&program.device_ctx)?[0];
        if error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "GPU postflight rejected transcript with code {error}"
            )));
        }
        let mut opcode_ranges = std::collections::BTreeMap::new();
        let mut covered = 0usize;
        for (&opcode, range) in program.active_opcodes.iter().zip(ranges.chunks_exact(2)) {
            let start = range[0] as usize;
            let end = range[1] as usize;
            if start != covered || start > end || end > num_steps {
                return Err(GpuRvrInputError::InvalidTranscript(
                    "GPU opcode ranges do not form a complete partition".to_string(),
                ));
            }
            if start != end {
                opcode_ranges.insert(opcode, start..end);
            }
            covered = end;
        }
        if covered != num_steps {
            return Err(GpuRvrInputError::InvalidTranscript(
                "GPU opcode ranges do not cover every execution step".to_string(),
            ));
        }
        Ok(Self {
            steps: steps_out,
            opcode_ranges,
            device_ctx: program.device_ctx.clone(),
            program_identity,
            segment_identity,
        })
    }

    pub fn steps(&self) -> DeviceBufferView {
        self.steps.view()
    }

    pub fn opcode_range(&self, opcode: VmOpcode) -> std::ops::Range<usize> {
        u32::try_from(opcode.as_usize())
            .ok()
            .and_then(|opcode| self.opcode_ranges.get(&opcode).cloned())
            .unwrap_or(0..0)
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn steps_host(&self) -> Result<Vec<[u32; 2]>, MemCopyError> {
        Ok(self
            .steps
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|step| [step.program_index, step.memory_start])
            .collect())
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn opcode_ranges_host(&self) -> &std::collections::BTreeMap<u32, std::ops::Range<usize>> {
        &self.opcode_ranges
    }
}

#[cfg(test)]
mod tests {
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        stream::GpuDeviceCtx,
    };
    use rvr_state::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
    };

    use super::*;

    fn event(
        timestamp: u32,
        address_space: u32,
        pointer: u32,
        is_write: bool,
    ) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: address_space | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
            pointer,
            value: [timestamp; 4],
        }
    }

    fn seed(address_space: u32, pointer: u32) -> PreflightInitialWrite {
        PreflightInitialWrite {
            address_space,
            pointer,
            initial_value: [0; 4],
        }
    }

    fn gpu_predecessors(
        memory: &[PreflightMemoryEvent],
        seeds: &[PreflightInitialWrite],
    ) -> (Vec<u32>, u32) {
        let config = MemoryConfig::default();
        gpu_predecessors_with_domain(
            memory,
            seeds,
            config.addr_space_height as u32,
            config.pointer_max_bits as u32,
        )
    }

    fn gpu_predecessors_with_domain(
        memory: &[PreflightMemoryEvent],
        seeds: &[PreflightInitialWrite],
        address_space_height: u32,
        pointer_max_bits: u32,
    ) -> (Vec<u32>, u32) {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let memory = upload(memory, &device_ctx).unwrap();
        let seeds = upload(seeds, &device_ctx).unwrap();
        let error = [0u32].to_device_on(&device_ctx).unwrap();
        let predecessors = build_gpu_memory_index(
            &memory,
            &seeds,
            address_space_height,
            pointer_max_bits,
            &error,
            &device_ctx,
        )
        .unwrap();
        (
            predecessors.to_host_on(&device_ctx).unwrap(),
            error.to_host_on(&device_ctx).unwrap()[0],
        )
    }

    fn gpu_program(opcodes: &[u32], device_ctx: &GpuDeviceCtx) -> GpuRvrProgram {
        let instructions = opcodes
            .iter()
            .map(|&opcode| RvrReplayInstruction {
                words: [opcode, 0, 0, 0, 0, 0, 0, 0],
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes.to_vec();
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        GpuRvrProgram {
            instructions: upload(&instructions, device_ctx).unwrap(),
            opcodes: opcodes.to_vec(),
            d_active_opcodes: upload(&active_opcodes, device_ctx).unwrap(),
            active_opcodes,
            address_space_height: MemoryConfig::default().addr_space_height as u32,
            pointer_max_bits: MemoryConfig::default().pointer_max_bits as u32,
            timestamp_max_bits: MemoryConfig::default().timestamp_max_bits as u32,
            pc_base: 0,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        }
    }

    fn gpu_plan(
        program: &GpuRvrProgram,
        transcript: &RvrPreflightTranscript,
        endpoint: RvrPreflightEndpoint,
    ) -> Result<GpuRvrReplayPlan, GpuRvrInputError> {
        let segment_identity = Arc::new(());
        let gpu_transcript = GpuRvrTranscript::upload(
            transcript,
            program.address_space_height,
            program.pointer_max_bits,
            &program.device_ctx,
            program.identity.clone(),
            segment_identity.clone(),
        )?;
        GpuRvrReplayPlan::build(
            program,
            &gpu_transcript,
            endpoint,
            program.identity.clone(),
            segment_identity,
        )
    }

    #[test]
    fn gpu_memory_index_matches_cpu_oracle() {
        let memory = vec![
            event(1, 1, 0, false),
            event(2, 3, 8, true),
            event(3, 1, 4, true),
            event(4, 2, 0, false),
            event(5, 1, 0, false),
            event(6, 3, 8, true),
        ];
        // Deliberately use an order different from the sorted block-key order.
        let seeds = vec![seed(3, 8), seed(1, 4)];
        let expected =
            super::super::postflight::build_memory_predecessors(&memory, &seeds).unwrap();
        let (actual, error) = gpu_predecessors(&memory, &seeds);
        assert_eq!(error, 0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn gpu_memory_index_rejects_invalid_seed_schedules() {
        let (_, missing) = gpu_predecessors(&[event(1, 1, 4, true)], &[]);
        assert_eq!(missing, 104);

        let duplicate_seed = seed(1, 4);
        let (_, duplicate) =
            gpu_predecessors(&[event(1, 1, 4, true)], &[duplicate_seed, duplicate_seed]);
        assert_eq!(duplicate, 105);

        let (_, unused) = gpu_predecessors(&[], &[seed(1, 4)]);
        assert_eq!(unused, 106);

        let (_, seed_before_read) = gpu_predecessors(&[event(1, 1, 4, false)], &[seed(1, 4)]);
        assert_eq!(seed_before_read, 106);
    }

    #[test]
    fn gpu_memory_index_rejects_non_increasing_timestamps() {
        let (_, error) = gpu_predecessors(&[event(1, 1, 0, false), event(1, 2, 0, false)], &[]);
        assert_eq!(error, 101);
    }

    #[test]
    fn gpu_memory_index_rejects_addresses_outside_its_bound_domain() {
        let config = MemoryConfig::default();
        let address_space_limit = ADDR_SPACE_OFFSET + (1 << config.addr_space_height);
        for invalid in [
            event(1, ADDR_SPACE_OFFSET - 1, 0, false),
            event(1, address_space_limit, 0, false),
            event(1, ADDR_SPACE_OFFSET, 2, false),
        ] {
            let (_, error) = gpu_predecessors(&[invalid], &[]);
            assert_eq!(error, 107);
        }

        let (_, error) =
            gpu_predecessors_with_domain(&[event(1, ADDR_SPACE_OFFSET, 16, false)], &[], 2, 4);
        assert_eq!(error, 107);
    }

    #[test]
    fn gpu_memory_index_accepts_the_maximum_compact_block_label() {
        let highest_address_space = ADDR_SPACE_OFFSET + 3;
        let (predecessors, error) = gpu_predecessors_with_domain(
            &[event(1, highest_address_space, u32::MAX - 3, false)],
            &[],
            2,
            32,
        );
        assert_eq!(error, 0);
        assert_eq!(predecessors, vec![0]);
    }

    #[test]
    fn gpu_program_rejects_memory_configs_outside_the_compact_key_abi() {
        use p3_baby_bear::BabyBear;

        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = Program::<BabyBear>::from_instructions(&[]);
        let assert_invalid = |config: &MemoryConfig| {
            assert!(matches!(
                GpuRvrProgram::upload(&program, config, &device_ctx),
                Err(GpuRvrInputError::InvalidMemoryConfig(_))
            ));
        };

        for pointer_max_bits in [1, 33] {
            let config = MemoryConfig {
                pointer_max_bits,
                ..MemoryConfig::default()
            };
            assert_invalid(&config);
        }

        let timestamp_too_wide = MemoryConfig {
            timestamp_max_bits: 32,
            ..MemoryConfig::default()
        };
        assert_invalid(&timestamp_too_wide);

        let label_too_wide = MemoryConfig {
            pointer_max_bits: 32,
            ..MemoryConfig::default()
        };
        assert_invalid(&label_too_wide);

        let mut malformed_layout = MemoryConfig::default();
        malformed_layout.addr_spaces.pop();
        assert_invalid(&malformed_layout);

        let mut maximum = MemoryConfig {
            addr_space_height: 2,
            pointer_max_bits: 32,
            ..MemoryConfig::default()
        };
        maximum
            .addr_spaces
            .truncate(ADDR_SPACE_OFFSET as usize + (1 << maximum.addr_space_height));
        GpuRvrProgram::upload(&program, &maximum, &device_ctx).unwrap();
    }

    #[test]
    fn gpu_program_index_matches_cpu_oracle_and_preserves_order() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let opcodes = [100, 200, terminate];
        let program = gpu_program(&opcodes, &device_ctx);
        let transcript = RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 2,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 3,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 4,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 5,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 5,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let endpoint = RvrPreflightEndpoint::Terminated;
        let expected =
            super::super::postflight::RvrReplayData::build(0, &opcodes, &transcript, endpoint)
                .unwrap();
        let actual = gpu_plan(&program, &transcript, endpoint).unwrap();
        let actual_steps = actual.steps.to_host_on(&device_ctx).unwrap();
        assert_eq!(actual_steps, expected.steps());
        assert_eq!(&actual.opcode_ranges, expected.opcode_ranges());
        assert_eq!(
            actual_steps[actual.opcode_ranges[&100].clone()]
                .iter()
                .map(|step| step.program_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            actual_steps[actual.opcode_ranges[&200].clone()]
                .iter()
                .map(|step| step.program_index)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn gpu_program_index_accepts_an_empty_suspended_segment() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = gpu_program(&[100], &device_ctx);
        let transcript = RvrPreflightTranscript {
            program_log: vec![PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            }],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let endpoint = RvrPreflightEndpoint::Suspended {
            resume_pc: 0,
            final_timestamp: 1,
        };
        let plan = gpu_plan(&program, &transcript, endpoint).unwrap();
        assert!(plan.steps.is_empty());
        assert!(plan.opcode_ranges.is_empty());
    }

    #[test]
    fn gpu_program_index_rejects_the_timestamp_domain_limit() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let mut program = gpu_program(&[100], &device_ctx);
        program.timestamp_max_bits = 2;
        let transcript = |final_timestamp| RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: final_timestamp,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        program
            .upload_transcript(
                &transcript(3),
                RvrPreflightEndpoint::Suspended {
                    resume_pc: 0,
                    final_timestamp: 3,
                },
            )
            .unwrap();
        assert!(program
            .upload_transcript(
                &transcript(4),
                RvrPreflightEndpoint::Suspended {
                    resume_pc: 0,
                    final_timestamp: 4,
                },
            )
            .is_err());
    }

    #[test]
    fn gpu_program_index_rejects_malformed_boundaries() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = gpu_program(&[100, terminate], &device_ctx);
        let transcript = |program_log| RvrPreflightTranscript {
            program_log,
            memory_log: vec![],
            initial_write_log: vec![],
        };

        let undefined_pc = transcript(vec![
            PreflightProgramEvent {
                pc: 12,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 12,
                timestamp: 2,
            },
        ]);
        assert!(gpu_plan(
            &program,
            &undefined_pc,
            RvrPreflightEndpoint::Suspended {
                resume_pc: 12,
                final_timestamp: 2,
            },
        )
        .is_err());

        let missing_terminate = transcript(vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0,
                timestamp: 2,
            },
        ]);
        assert!(gpu_plan(
            &program,
            &missing_terminate,
            RvrPreflightEndpoint::Terminated,
        )
        .is_err());

        let timestamp_regression = transcript(vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 1,
            },
        ]);
        assert!(gpu_plan(
            &program,
            &timestamp_regression,
            RvrPreflightEndpoint::Terminated,
        )
        .is_err());
    }
}
