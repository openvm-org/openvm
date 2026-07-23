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
use p3_baby_bear::BabyBear;
use rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent};
use thiserror::Error;

#[cfg(feature = "test-utils")]
use super::postflight::RvrReplayData;
use super::{postflight::RvrReplayStep, RvrPreflightEndpoint, RvrPreflightTranscript};
use crate::{
    arch::{ExecutionState, MemoryCellType, MemoryConfig, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    cuda_abi::rvr_postflight,
    system::TouchedBlock,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RvrReplayInstruction {
    /// Global opcode followed by the seven canonical instruction operands.
    pub words: [u32; 8],
}

const _: () = assert!(size_of::<RvrReplayInstruction>() == 32);
const _: () = assert!(size_of::<TouchedBlock<BabyBear>>() == 7 * size_of::<u32>());

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RvrMemoryAddressSpace {
    num_cells: u64,
    is_u16: u32,
    _padding: u32,
}

const _: () = assert!(size_of::<RvrMemoryAddressSpace>() == 16);

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

pub(crate) type ConnectorBoundary = (ExecutionState<u32>, ExecutionState<u32>, Option<u32>);

fn replay_boundary(
    transcript: &RvrPreflightTranscript,
    endpoint: RvrPreflightEndpoint,
) -> Result<ConnectorBoundary, GpuRvrInputError> {
    let first = transcript.program_log.first().ok_or_else(|| {
        GpuRvrInputError::InvalidTranscript(
            "transcript must contain an initial event and final sentinel".to_string(),
        )
    })?;
    let last = transcript.program_log.last().unwrap();
    Ok((
        ExecutionState::new(first.pc, first.timestamp),
        ExecutionState::new(last.pc, last.timestamp),
        matches!(endpoint, RvrPreflightEndpoint::Terminated).then_some(0),
    ))
}

/// Static program data uploaded once and shared by every replayed segment.
pub struct GpuRvrProgram {
    instructions: DeviceBuffer<RvrReplayInstruction>,
    dense_program_rows: DeviceBuffer<u32>,
    num_program_rows: usize,
    #[cfg(feature = "test-utils")]
    opcodes: Vec<u32>,
    active_opcodes: Vec<u32>,
    d_active_opcodes: DeviceBuffer<u32>,
    memory_address_spaces: DeviceBuffer<RvrMemoryAddressSpace>,
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
        if F::ORDER_U32 != BabyBear::ORDER_U32 || size_of::<F>() != size_of::<BabyBear>() {
            return Err(GpuRvrInputError::InvalidMemoryConfig(
                "RVR GPU postflight currently requires the BabyBear proof field".to_string(),
            ));
        }
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
        let memory_address_spaces = memory_config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                is_u16: u32::from(config.layout == MemoryCellType::U16),
                _padding: 0,
            })
            .collect::<Vec<_>>();
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
        let mut next_program_row = 0u32;
        let dense_program_rows = opcodes
            .iter()
            .map(|&opcode| {
                if opcode == u32::MAX {
                    u32::MAX
                } else {
                    let row = next_program_row;
                    next_program_row += 1;
                    row
                }
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes
            .iter()
            .copied()
            .filter(|&opcode| opcode != u32::MAX)
            .collect::<Vec<_>>();
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            dense_program_rows: upload(&dense_program_rows, device_ctx)?,
            num_program_rows: next_program_row as usize,
            #[cfg(feature = "test-utils")]
            opcodes,
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx)?,
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
        let boundary = replay_boundary(transcript, endpoint)?;
        let segment_identity = Arc::new(());
        let gpu = GpuRvrTranscript::upload(
            transcript,
            self.address_space_height,
            self.pointer_max_bits,
            self.memory_address_spaces.view(),
            &self.device_ctx,
            self.identity.clone(),
            segment_identity.clone(),
        )?;
        let plan = GpuRvrReplayPlan::build(
            self,
            &gpu,
            endpoint,
            boundary,
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
        let boundary = replay_boundary(transcript, endpoint)?;
        self.device_ctx.stream.synchronize()?;
        let started = Instant::now();
        let segment_identity = Arc::new(());
        let transcript = GpuRvrTranscript::upload(
            transcript,
            self.address_space_height,
            self.pointer_max_bits,
            self.memory_address_spaces.view(),
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
            boundary,
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
        let memory_scatter_stage =
            16 * num_entries + 32 * num_memory + memory_temp + size_of::<u32>();
        let program_stage = 32 * num_memory
            + 24 * num_steps
            + 8 * self.active_opcodes.len()
            + 4 * self.num_program_rows
            + program_temp;
        let steady = 32 * num_memory + 8 * num_steps + 4 * self.num_program_rows;
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

struct GpuMemoryIndex {
    predecessors: DeviceBuffer<u32>,
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
}

fn build_gpu_memory_index(
    memory: &DeviceBuffer<PreflightMemoryEvent>,
    seeds: &DeviceBuffer<PreflightInitialWrite>,
    address_space_height: u32,
    pointer_max_bits: u32,
    address_spaces: DeviceBufferView,
    error: &DeviceBuffer<u32>,
    device_ctx: &GpuDeviceCtx,
) -> Result<GpuMemoryIndex, GpuRvrInputError> {
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
            address_spaces,
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
    // The scan reuses the sort scratch. All allocations and frees remain
    // ordered on this stream.
    drop(keys_in);
    let memory_predecessors = gpu_buffer::<u32>(memory.len(), device_ctx);
    let touched_flags = gpu_buffer::<u32>(num_entries, device_ctx);
    let touched_positions = gpu_buffer::<u32>(num_entries, device_ctx);
    let touched_blocks = gpu_buffer::<TouchedBlock<BabyBear>>(memory.len(), device_ctx);
    let num_touched_blocks = [0u32].to_device_on(device_ctx)?;
    unsafe {
        rvr_postflight::memory_index_scatter(
            memory.view(),
            seeds.len(),
            &keys_out,
            num_entries,
            &memory_predecessors,
            &touched_flags,
            &touched_positions,
            touched_blocks.as_mut_raw_ptr(),
            &num_touched_blocks,
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    let num_touched_blocks = num_touched_blocks.to_host_on(device_ctx)?[0] as usize;
    if num_touched_blocks > memory.len() {
        return Err(GpuRvrInputError::InvalidTranscript(
            "GPU touched-block count exceeds the memory log".to_string(),
        ));
    }
    Ok(GpuMemoryIndex {
        predecessors: memory_predecessors,
        touched_blocks,
        num_touched_blocks,
    })
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
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
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
        address_spaces: DeviceBufferView,
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
        let memory_index = build_gpu_memory_index(
            &memory_log,
            &initial_write_log,
            address_space_height,
            pointer_max_bits,
            address_spaces,
            &error,
            device_ctx,
        )?;
        Ok(Self {
            program_log,
            memory_log,
            initial_write_log,
            memory_predecessors: memory_index.predecessors,
            touched_blocks: memory_index.touched_blocks,
            num_touched_blocks: memory_index.num_touched_blocks,
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

    /// Sorted unique final state of every block touched by a timed memory
    /// event. The view is the initialized prefix of the retained capacity.
    pub fn touched_blocks(&self) -> DeviceBufferView {
        DeviceBufferView {
            ptr: self.touched_blocks.as_raw_ptr(),
            size: self.num_touched_blocks * size_of::<TouchedBlock<BabyBear>>(),
        }
    }

    pub const fn num_touched_blocks(&self) -> usize {
        self.num_touched_blocks
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn memory_predecessors_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.memory_predecessors.to_host_on(&self.device_ctx)
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn touched_blocks_host(&self) -> Result<Vec<TouchedBlock<BabyBear>>, MemCopyError> {
        let mut blocks = self.touched_blocks.to_host_on(&self.device_ctx)?;
        blocks.truncate(self.num_touched_blocks);
        Ok(blocks)
    }

    pub fn error_ptr(&self) -> *mut u32 {
        self.error.as_mut_ptr()
    }
}

/// The opcode-partitioned replay work list, uploaded once per segment.
pub struct GpuRvrReplayPlan {
    steps: DeviceBuffer<RvrReplayStep>,
    program_frequencies: DeviceBuffer<u32>,
    opcode_ranges: std::collections::BTreeMap<u32, std::ops::Range<usize>>,
    from_state: ExecutionState<u32>,
    to_state: ExecutionState<u32>,
    exit_code: Option<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuRvrReplayPlan {
    fn build(
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        endpoint: RvrPreflightEndpoint,
        boundary: ConnectorBoundary,
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
        let program_frequencies =
            upload(&vec![0u32; program.num_program_rows], &program.device_ctx)?;
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
                program.dense_program_rows.view(),
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
                &program_frequencies,
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
            program_frequencies,
            opcode_ranges,
            from_state: boundary.0,
            to_state: boundary.1,
            exit_code: boundary.2,
            device_ctx: program.device_ctx.clone(),
            program_identity,
            segment_identity,
        })
    }

    pub fn steps(&self) -> DeviceBufferView {
        self.steps.view()
    }

    /// Dense execution frequencies in cached-program row order. Static program
    /// gaps are omitted and unexecuted defined instructions remain zero.
    pub fn program_frequencies(&self) -> DeviceBufferView {
        self.program_frequencies.view()
    }

    /// Connector inputs derived from the same host events uploaded into this
    /// validated replay plan. This metadata is cold and adds nothing to the
    /// preflight hot-path logs.
    pub(crate) const fn connector_boundary(&self) -> ConnectorBoundary {
        (self.from_state, self.to_state, self.exit_code)
    }

    pub fn opcode_range(&self, opcode: VmOpcode) -> std::ops::Range<usize> {
        u32::try_from(opcode.as_usize())
            .ok()
            .and_then(|opcode| self.opcode_ranges.get(&opcode).cloned())
            .unwrap_or(0..0)
    }

    /// Global opcodes that were actually executed in this segment.
    ///
    /// The iterator is sorted and contains no duplicates because it follows
    /// the replay plan's opcode partition. Tracegen coordinators use this
    /// before launching any opcode kernel to reject unported instructions.
    pub fn executed_opcodes(&self) -> impl Iterator<Item = u32> + '_ {
        self.opcode_ranges.keys().copied()
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
    pub fn program_frequencies_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.program_frequencies.to_host_on(&self.device_ctx)
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

    fn event_value(
        timestamp: u32,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        value: [u32; 4],
    ) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: address_space | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
            pointer,
            value,
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
        let pointer_limit = 1u64 << pointer_max_bits;
        let address_spaces = upload(
            &vec![
                RvrMemoryAddressSpace {
                    num_cells: pointer_limit,
                    is_u16: 1,
                    _padding: 0,
                };
                ADDR_SPACE_OFFSET as usize + (1usize << address_space_height)
            ],
            &device_ctx,
        )
        .unwrap();
        let error = [0u32].to_device_on(&device_ctx).unwrap();
        let index = build_gpu_memory_index(
            &memory,
            &seeds,
            address_space_height,
            pointer_max_bits,
            address_spaces.view(),
            &error,
            &device_ctx,
        )
        .unwrap();
        (
            index.predecessors.to_host_on(&device_ctx).unwrap(),
            error.to_host_on(&device_ctx).unwrap()[0],
        )
    }

    fn gpu_memory_index_with_config(
        memory: &[PreflightMemoryEvent],
        seeds: &[PreflightInitialWrite],
        config: &MemoryConfig,
    ) -> (Vec<u32>, Vec<TouchedBlock<BabyBear>>, u32) {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let memory = upload(memory, &device_ctx).unwrap();
        let seeds = upload(seeds, &device_ctx).unwrap();
        let address_spaces = config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                is_u16: u32::from(config.layout == MemoryCellType::U16),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        let address_spaces = upload(&address_spaces, &device_ctx).unwrap();
        let error = [0u32].to_device_on(&device_ctx).unwrap();
        let index = build_gpu_memory_index(
            &memory,
            &seeds,
            config.addr_space_height as u32,
            config.pointer_max_bits as u32,
            address_spaces.view(),
            &error,
            &device_ctx,
        )
        .unwrap();
        let mut touched = index.touched_blocks.to_host_on(&device_ctx).unwrap();
        touched.truncate(index.num_touched_blocks);
        (
            index.predecessors.to_host_on(&device_ctx).unwrap(),
            touched,
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
        active_opcodes.retain(|&opcode| opcode != u32::MAX);
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        let mut next_program_row = 0u32;
        let dense_program_rows = opcodes
            .iter()
            .map(|&opcode| {
                if opcode == u32::MAX {
                    u32::MAX
                } else {
                    let row = next_program_row;
                    next_program_row += 1;
                    row
                }
            })
            .collect::<Vec<_>>();
        let config = MemoryConfig::default();
        let memory_address_spaces = config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                is_u16: u32::from(config.layout == MemoryCellType::U16),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        GpuRvrProgram {
            instructions: upload(&instructions, device_ctx).unwrap(),
            dense_program_rows: upload(&dense_program_rows, device_ctx).unwrap(),
            num_program_rows: next_program_row as usize,
            opcodes: opcodes.to_vec(),
            d_active_opcodes: upload(&active_opcodes, device_ctx).unwrap(),
            active_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx).unwrap(),
            address_space_height: config.addr_space_height as u32,
            pointer_max_bits: config.pointer_max_bits as u32,
            timestamp_max_bits: config.timestamp_max_bits as u32,
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
        let boundary = replay_boundary(transcript, endpoint)?;
        let segment_identity = Arc::new(());
        let gpu_transcript = GpuRvrTranscript::upload(
            transcript,
            program.address_space_height,
            program.pointer_max_bits,
            program.memory_address_spaces.view(),
            &program.device_ctx,
            program.identity.clone(),
            segment_identity.clone(),
        )?;
        GpuRvrReplayPlan::build(
            program,
            &gpu_transcript,
            endpoint,
            boundary,
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
    fn gpu_touched_blocks_match_tracing_memory_finalize() {
        use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};

        use crate::system::memory::online::TracingMemory;

        let config = MemoryConfig::default();
        let memory = vec![
            event_value(1, RV64_MEMORY_AS, 4, false, [0; 4]),
            event_value(2, RV64_REGISTER_AS, 4, true, [1, u16::MAX as u32, 0, 3]),
            event_value(3, RV64_MEMORY_AS, 0, true, [10, 11, 12, 13]),
            event_value(4, RV64_MEMORY_AS, 4, false, [0; 4]),
            event_value(5, RV64_MEMORY_AS, 4, true, [20, 21, 22, 23]),
            event_value(6, RV64_MEMORY_AS, 4, false, [20, 21, 22, 23]),
            event_value(7, RV64_MEMORY_AS, 0, false, [10, 11, 12, 13]),
        ];
        // Deliberately opposite the final touched-block sort order.
        let seeds = vec![
            PreflightInitialWrite {
                address_space: RV64_MEMORY_AS,
                pointer: 0,
                initial_value: [0; 4],
            },
            PreflightInitialWrite {
                address_space: RV64_REGISTER_AS,
                pointer: 4,
                initial_value: [0; 4],
            },
        ];

        let mut cpu = TracingMemory::new(&config);
        unsafe {
            cpu.read::<u16, 4>(RV64_MEMORY_AS, 4);
            cpu.write::<u16, 4>(RV64_REGISTER_AS, 4, [1, u16::MAX, 0, 3]);
            cpu.write::<u16, 4>(RV64_MEMORY_AS, 0, [10, 11, 12, 13]);
            cpu.read::<u16, 4>(RV64_MEMORY_AS, 4);
            cpu.write::<u16, 4>(RV64_MEMORY_AS, 4, [20, 21, 22, 23]);
            cpu.read::<u16, 4>(RV64_MEMORY_AS, 4);
            cpu.read::<u16, 4>(RV64_MEMORY_AS, 0);
        }
        let expected = cpu.finalize::<BabyBear>();
        let (_, actual, error) = gpu_memory_index_with_config(&memory, &seeds, &config);
        assert_eq!(error, 0);
        assert_eq!(actual, expected);
        assert_eq!(
            actual
                .iter()
                .map(|block| (block.address_space, block.ptr, block.timestamp))
                .collect::<Vec<_>>(),
            vec![
                (RV64_REGISTER_AS, 4, 2),
                (RV64_MEMORY_AS, 0, 7),
                (RV64_MEMORY_AS, 4, 6),
            ]
        );
    }

    #[test]
    fn gpu_touched_blocks_accept_empty_and_last_aligned_u16_block() {
        use openvm_instructions::riscv::RV64_REGISTER_AS;

        let config = MemoryConfig::default();
        let (_, touched, error) = gpu_memory_index_with_config(&[], &[], &config);
        assert_eq!(error, 0);
        assert!(touched.is_empty());

        let pointer = config.addr_spaces[RV64_REGISTER_AS as usize].num_cells as u32 - 4;
        let (_, touched, error) = gpu_memory_index_with_config(
            &[event_value(
                1,
                RV64_REGISTER_AS,
                pointer,
                false,
                [u16::MAX as u32; 4],
            )],
            &[],
            &config,
        );
        assert_eq!(error, 0);
        assert_eq!(touched.len(), 1);
        assert_eq!(touched[0].ptr, pointer);
        assert_eq!(
            touched[0].values.map(|value| value.as_canonical_u32()),
            [u16::MAX as u32; 4]
        );
    }

    #[test]
    fn gpu_memory_metadata_fails_closed_for_bad_layout_bounds_and_values() {
        use openvm_instructions::{riscv::RV64_REGISTER_AS, DEFERRAL_AS};

        let config = MemoryConfig::default();
        let assert_rejected =
            |memory: &[PreflightMemoryEvent], seeds: &[PreflightInitialWrite], config| {
                let (_, _, error) = gpu_memory_index_with_config(memory, seeds, config);
                assert_ne!(error, 0);
            };

        assert_rejected(
            &[event_value(1, DEFERRAL_AS, 0, false, [0; 4])],
            &[],
            &config,
        );
        assert_rejected(
            &[event_value(
                1,
                RV64_REGISTER_AS,
                0,
                false,
                [0, 0, 0, u16::MAX as u32 + 1],
            )],
            &[],
            &config,
        );
        assert_rejected(
            &[event_value(1, RV64_REGISTER_AS, 0, true, [0; 4])],
            &[PreflightInitialWrite {
                address_space: RV64_REGISTER_AS,
                pointer: 0,
                initial_value: [0, 0, 0, u16::MAX as u32 + 1],
            }],
            &config,
        );
        let end = config.addr_spaces[RV64_REGISTER_AS as usize].num_cells as u32;
        assert_rejected(
            &[event_value(1, RV64_REGISTER_AS, end, false, [0; 4])],
            &[],
            &config,
        );
        let mut crossing = config.clone();
        crossing.addr_spaces[RV64_REGISTER_AS as usize].num_cells -= 2;
        assert_rejected(
            &[event_value(1, RV64_REGISTER_AS, end - 4, false, [0; 4])],
            &[],
            &crossing,
        );

        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = gpu_program(&[100], &device_ctx);
        let malformed = RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 2,
                },
            ],
            memory_log: vec![event_value(
                1,
                RV64_REGISTER_AS,
                0,
                false,
                [0, 0, 0, u16::MAX as u32 + 1],
            )],
            initial_write_log: vec![],
        };
        assert!(program
            .upload_transcript(
                &malformed,
                RvrPreflightEndpoint::Suspended {
                    resume_pc: 0,
                    final_timestamp: 2,
                },
            )
            .is_err());

        let malformed_seed = RvrPreflightTranscript {
            program_log: malformed.program_log.clone(),
            memory_log: vec![event_value(1, RV64_REGISTER_AS, 0, true, [0; 4])],
            initial_write_log: vec![PreflightInitialWrite {
                address_space: RV64_REGISTER_AS,
                pointer: 0,
                initial_value: [0, 0, 0, u16::MAX as u32 + 1],
            }],
        };
        assert!(program
            .upload_transcript(
                &malformed_seed,
                RvrPreflightEndpoint::Suspended {
                    resume_pc: 0,
                    final_timestamp: 2,
                },
            )
            .is_err());
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
    fn gpu_program_frequencies_are_dense_and_exclude_the_sentinel() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let mut program = gpu_program(&[100, u32::MAX, 200, 300, terminate], &device_ctx);
        program.pc_base = 0x100;
        let transcript = RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0x108,
                    timestamp: 2,
                },
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 3,
                },
                PreflightProgramEvent {
                    pc: 0x110,
                    timestamp: 4,
                },
                PreflightProgramEvent {
                    pc: 0x110,
                    timestamp: 4,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(&program, &transcript, RvrPreflightEndpoint::Terminated).unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![2, 1, 0, 1]
        );
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x110u32, 4u32),
                Some(0)
            )
        );

        let suspended = RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0x108,
                    timestamp: 2,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(
            &program,
            &suspended,
            RvrPreflightEndpoint::Suspended {
                resume_pc: 0x108,
                final_timestamp: 2,
            },
        )
        .unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![1, 0, 0, 0]
        );
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x108u32, 2u32),
                None
            )
        );

        let empty = RvrPreflightTranscript {
            program_log: vec![PreflightProgramEvent {
                pc: 0x100,
                timestamp: 1,
            }],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(
            &program,
            &empty,
            RvrPreflightEndpoint::Suspended {
                resume_pc: 0x100,
                final_timestamp: 1,
            },
        )
        .unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![0; 4]
        );
    }

    #[test]
    fn gpu_program_frequency_input_rejects_invalid_program_counters() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let mut program = gpu_program(&[100, u32::MAX, 200], &device_ctx);
        program.pc_base = 0x100;
        for invalid_pc in [0xfc, 0x102, 0x104, 0x10c] {
            let transcript = RvrPreflightTranscript {
                program_log: vec![
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 1,
                    },
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 2,
                    },
                ],
                memory_log: vec![],
                initial_write_log: vec![],
            };
            assert!(gpu_plan(
                &program,
                &transcript,
                RvrPreflightEndpoint::Suspended {
                    resume_pc: invalid_pc,
                    final_timestamp: 2,
                },
            )
            .is_err());
        }
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
