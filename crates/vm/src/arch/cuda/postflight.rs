//! GPU indexing and read-only replay for immutable preflight history.
//!
//! This module uploads static program metadata, builds memory chronology and
//! opcode indexes once per segment, and exposes immutable replay inputs to
//! system and instruction trace generators. Compiled checkpoint expansion is
//! an optional producer supplied by an execution extension; this layer does
//! not depend on an instruction set.

use std::{
    collections::BTreeMap,
    mem::{align_of, offset_of, size_of},
    ops::Range,
    sync::Arc,
};

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::{CudaError, MemCopyError},
    memory_manager::MemTracker,
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    instruction::Instruction, program::Program, LocalOpcode, SystemOpcode, VmOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
};
use thiserror::Error;
use tracing::{info_span, span::EnteredSpan};

use crate::{
    arch::{
        postflight::validate_postflight_memory_config, AddressSpaceHostLayout, ExecutionState,
        MemoryCellType, MemoryConfig, PreflightHistory, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH,
        POSTFLIGHT_PREDECESSOR_INDEX_LIMIT,
    },
    cuda_abi::postflight,
    system::TouchedBlock,
};

#[cfg(all(test, feature = "rvr"))]
mod integration_tests;
#[cfg(any(test, feature = "test-utils"))]
mod testing;

/// Number of `u32` fields in a replay instruction: global opcode followed by operands `a..g`.
pub const POSTFLIGHT_INSTRUCTION_FIELDS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuReplayInstruction {
    /// Global opcode followed by the seven canonical instruction operands.
    pub words: [u32; POSTFLIGHT_INSTRUCTION_FIELDS],
}

/// Location of one executed instruction and the first timed memory event in
/// its timestamp interval. The final program sentinel has no entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GpuReplayStep {
    pub program_index: u32,
    pub memory_start: u32,
}

// Keep Rust uploads byte-compatible with the CUDA replay instruction ABI.
const _: () = assert!(size_of::<GpuReplayInstruction>() == size_of::<[u32; 8]>());

/// Device-side layout metadata for one configured address space.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuMemoryAddressSpace {
    num_cells: u64,
    cell_kind: GpuMemoryCellKind,
    _padding: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuMemoryCellKind {
    Unsupported = 0,
    U16 = 1,
    Field32 = 2,
}

#[derive(Clone, Copy, Debug)]
struct GpuMemoryDimensions {
    addr_space_height: u32,
    cell_pointer_max_bits: u32,
    block_pointer_max_bits: u32,
    timestamp_max_bits: u32,
}

impl GpuMemoryDimensions {
    fn from_validated(config: &MemoryConfig) -> Self {
        let block_pointer_bits = BLOCK_FE_WIDTH.ilog2() as usize;
        Self {
            addr_space_height: config.addr_space_height as u32,
            cell_pointer_max_bits: config.pointer_max_bits as u32,
            block_pointer_max_bits: (config.pointer_max_bits - block_pointer_bits) as u32,
            timestamp_max_bits: config.timestamp_max_bits as u32,
        }
    }
}

// Keep address-space metadata byte-compatible with its CUDA mirror.
const _: () = {
    assert!(size_of::<GpuMemoryCellKind>() == size_of::<u32>());
    assert!(size_of::<GpuMemoryAddressSpace>() == 16);
    assert!(align_of::<GpuMemoryAddressSpace>() == align_of::<u64>());
    assert!(offset_of!(GpuMemoryAddressSpace, num_cells) == 0);
    assert!(offset_of!(GpuMemoryAddressSpace, cell_kind) == 8);
    assert!(offset_of!(GpuMemoryAddressSpace, _padding) == 12);
};

fn memory_cell_kind(layout: MemoryCellType) -> GpuMemoryCellKind {
    match layout {
        MemoryCellType::U16 => GpuMemoryCellKind::U16,
        MemoryCellType::FIELD32 => GpuMemoryCellKind::Field32,
        _ => GpuMemoryCellKind::Unsupported,
    }
}

fn validate_field_address_spaces(memory_config: &MemoryConfig) -> Result<(), GpuPostflightError> {
    if let Some((address_space, _)) =
        memory_config
            .addr_spaces
            .iter()
            .enumerate()
            .find(|(address_space, config)| {
                config.num_cells != 0
                    && config.layout == MemoryCellType::field32()
                    && *address_space != DEFERRAL_AS as usize
            })
    {
        return Err(GpuPostflightError::InvalidMemoryConfig(format!(
            "field-cell address space {address_space} is unsupported; only DEFERRAL_AS \
             ({DEFERRAL_AS}) may use 4-byte field cells"
        )));
    }
    Ok(())
}

fn validate_initial_memory_lengths(
    memory_config: &MemoryConfig,
    byte_lengths: &[usize],
) -> Result<(), GpuPostflightError> {
    if byte_lengths.len() != memory_config.addr_spaces.len() {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "initial-memory table has {} address spaces, expected {}",
            byte_lengths.len(),
            memory_config.addr_spaces.len()
        )));
    }
    for (address_space, (&actual_bytes, config)) in byte_lengths
        .iter()
        .zip(&memory_config.addr_spaces)
        .enumerate()
    {
        let expected_bytes = config
            .num_cells
            .checked_mul(config.layout.size())
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig(format!(
                    "address space {address_space} byte length overflows usize"
                ))
            })?;
        if actual_bytes != expected_bytes {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "initial-memory address space {address_space} has {actual_bytes} bytes, expected \
                 {expected_bytes}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum GpuPostflightError {
    #[error("program opcode {0} does not fit the GPU replay ABI")]
    OpcodeTooLarge(usize),
    #[error("invalid GPU postflight memory configuration: {0}")]
    InvalidMemoryConfig(String),
    #[error("invalid GPU postflight access schedule: {0}")]
    InvalidAccessSchedule(String),
    #[error("{0}")]
    InvalidTranscript(String),
    #[error("GPU postflight input belongs to another CUDA device or stream")]
    ContextMismatch,
    #[error("GPU postflight transcript belongs to another uploaded program")]
    ProgramMismatch,
    #[error("GPU postflight plan belongs to another transcript segment")]
    SegmentMismatch,
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Copy(#[from] MemCopyError),
}

pub(crate) fn upload<T>(
    values: &[T],
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<T>, MemCopyError> {
    if values.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        values.to_device_on(device_ctx)
    }
}

pub(crate) type ConnectorBoundary = (ExecutionState<u32>, ExecutionState<u32>, Option<u32>);

/// Architectural boundary attached to one device-resident preflight history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPostflightBoundary {
    from: ExecutionState<u32>,
    to: ExecutionState<u32>,
    exit_code: Option<u32>,
}

impl GpuPostflightBoundary {
    pub const fn new(
        from: ExecutionState<u32>,
        to: ExecutionState<u32>,
        exit_code: Option<u32>,
    ) -> Self {
        Self {
            from,
            to,
            exit_code,
        }
    }

    const fn connector_boundary(self) -> ConnectorBoundary {
        (self.from, self.to, self.exit_code)
    }
}

/// Device-resident program and memory history before chronology and opcode indexing.
///
/// Producers own how these append-only logs are built. Finalization has one
/// implementation in [`GpuPostflightContext::finalize_device_history`].
pub struct GpuUnindexedHistory {
    program: DeviceBuffer<PreflightProgramEvent>,
    memory: DeviceBuffer<PreflightMemoryEvent>,
    field_values: DeviceBuffer<PreflightFieldBlock>,
    write_masks: DeviceBuffer<u8>,
    error: DeviceBuffer<u32>,
}

impl GpuUnindexedHistory {
    pub fn new(
        program: DeviceBuffer<PreflightProgramEvent>,
        memory: DeviceBuffer<PreflightMemoryEvent>,
        field_values: DeviceBuffer<PreflightFieldBlock>,
        write_masks: DeviceBuffer<u8>,
        error: DeviceBuffer<u32>,
    ) -> Result<Self, GpuPostflightError> {
        if write_masks.len() != memory.len() {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "device history has {} memory events but {} write masks",
                memory.len(),
                write_masks.len()
            )));
        }
        if error.len() != 1 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "device history error buffer has {} elements, expected one",
                error.len()
            )));
        }
        Ok(Self {
            program,
            memory,
            field_values,
            write_masks,
            error,
        })
    }
}

/// Validated access to the fixed program and segment-start memory used by GPU postflight.
///
/// The borrowed initial-memory owners remain alive until all producer work and
/// common chronology/indexing have been submitted and synchronized.
pub struct GpuPostflightContext<'a> {
    program: &'a GpuPostflightProgram,
    initial_memory: &'a [Arc<DeviceBuffer<u8>>],
    memory_metrics: MemTracker,
    _span: EnteredSpan,
}

pub struct GpuPostflightProgram {
    instructions: DeviceBuffer<GpuReplayInstruction>,
    dense_program_rows: DeviceBuffer<u32>,
    num_program_rows: usize,
    active_opcodes: Vec<u32>,
    d_active_opcodes: DeviceBuffer<u32>,
    memory_address_spaces: DeviceBuffer<GpuMemoryAddressSpace>,
    /// Host layout for validating and uploading expanded interpreter logs.
    memory_config: MemoryConfig,
    address_space_height: u32,
    /// Pointer width used by chronology keys, whose pointers count AS-native cells.
    cell_pointer_max_bits: u32,
    timestamp_max_bits: u32,
    pc_base: u32,
    device_ctx: GpuDeviceCtx,
    identity: Arc<()>,
}

impl<'a> GpuPostflightContext<'a> {
    pub(crate) fn new(
        program: &'a GpuPostflightProgram,
        program_device_ctx: &GpuDeviceCtx,
        memory_device_ctx: &GpuDeviceCtx,
        initial_memory: &'a [Arc<DeviceBuffer<u8>>],
    ) -> Result<Self, GpuPostflightError> {
        program.validate_system_inputs(program_device_ctx, memory_device_ctx, initial_memory)?;
        Ok(Self {
            program,
            initial_memory,
            memory_metrics: MemTracker::start_and_reset_peak("postflight"),
            _span: info_span!("postflight").entered(),
        })
    }

    /// Returns the immutable segment-start image for one configured address space.
    pub fn memory_image(&self, address_space: u32) -> Result<DeviceBufferView, GpuPostflightError> {
        self.initial_memory
            .get(address_space as usize)
            .map(|image| image.view())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(format!(
                    "initial-memory address space {address_space} was not transported to the GPU"
                ))
            })
    }

    /// Runs common memory chronology and opcode indexing over producer-owned device logs.
    pub fn finalize_device_history(
        self,
        history: GpuUnindexedHistory,
        boundary: GpuPostflightBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let initial_memory_images = self
            .initial_memory
            .iter()
            .map(|image| image.view())
            .collect::<Vec<_>>();
        self.program.index_device_history(
            history,
            &initial_memory_images,
            boundary.connector_boundary(),
        )
    }

    pub(crate) fn upload_history(
        self,
        history: &PreflightHistory,
        boundary: GpuPostflightBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let initial_memory_images = self
            .initial_memory
            .iter()
            .map(|image| image.view())
            .collect::<Vec<_>>();
        self.program.upload_history(
            history,
            boundary.connector_boundary(),
            &initial_memory_images,
        )
    }
}

impl Drop for GpuPostflightContext<'_> {
    fn drop(&mut self) {
        self.memory_metrics.emit_metrics();
    }
}

fn validated_history_write_masks(
    history: &PreflightHistory,
    memory_config: &MemoryConfig,
) -> Result<Vec<u8>, GpuPostflightError> {
    if history.program.is_empty() {
        return Err(GpuPostflightError::InvalidTranscript(
            "preflight history must contain a final program sentinel".to_string(),
        ));
    }
    if history.memory.accesses.len() >= POSTFLIGHT_PREDECESSOR_INDEX_LIMIT as usize {
        return Err(GpuPostflightError::InvalidTranscript(
            "preflight memory log exceeds packed predecessor indexes".to_string(),
        ));
    }

    let mut write_masks = Vec::with_capacity(history.memory.accesses.len());
    let mut field_cursor = 0usize;
    for event in &history.memory.accesses {
        let address_space = event.address_space() as usize;
        let layout = memory_config
            .addr_spaces
            .get(address_space)
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(format!(
                    "memory event uses unknown address space {address_space}"
                ))
            })?
            .layout;
        write_masks.push(if event.is_write() { u8::MAX } else { 0 });
        if layout == MemoryCellType::field32() {
            let reference =
                usize::try_from(u32::from(event.value[0]) | (u32::from(event.value[1]) << 16))
                    .unwrap();
            if event.value[2] != 0
                || event.value[3] != 0
                || reference != field_cursor
                || reference >= history.memory.field_values.len()
            {
                return Err(GpuPostflightError::InvalidTranscript(
                    "field memory events must use dense ordered sidecar references".to_string(),
                ));
            }
            field_cursor += 1;
        }
    }
    if field_cursor != history.memory.field_values.len() {
        return Err(GpuPostflightError::InvalidTranscript(
            "field sidecar contains unreferenced values".to_string(),
        ));
    }
    Ok(write_masks)
}

impl GpuPostflightProgram {
    /// Uploads the immutable program metadata used by history-driven
    /// postflight and trace generation.
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        Self::upload_with_instruction_validation(program, memory_config, device_ctx, |_| Ok(()))
    }

    /// Uploads fixed program data after producer-owned instruction validation.
    ///
    /// The validator sees the exact canonical instruction representation that
    /// is uploaded, allowing an extension-owned replay producer to reject
    /// unsupported instruction layouts without moving ISA semantics into the VM.
    pub fn upload_with_instruction_validation<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
        mut validate_instruction: impl FnMut(&GpuReplayInstruction) -> Result<(), GpuPostflightError>,
    ) -> Result<Self, GpuPostflightError> {
        if F::ORDER_U32 != BabyBear::ORDER_U32 || size_of::<F>() != size_of::<BabyBear>() {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "GPU postflight currently requires the BabyBear proof field".to_string(),
            ));
        }
        validate_postflight_memory_config(memory_config)
            .map_err(GpuPostflightError::InvalidMemoryConfig)?;
        let dimensions = GpuMemoryDimensions::from_validated(memory_config);
        validate_field_address_spaces(memory_config)?;
        let memory_address_spaces = memory_config
            .addr_spaces
            .iter()
            .map(|config| GpuMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_kind: memory_cell_kind(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        if dimensions.addr_space_height + dimensions.block_pointer_max_bits > u32::BITS {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "address-space and block-pointer label does not fit u32".to_string(),
            ));
        }
        let instructions = program
            .instructions_and_debug_infos
            .iter()
            .map(|entry| match entry {
                Some((instruction, _)) => instruction_to_replay(instruction),
                None => Ok(GpuReplayInstruction {
                    words: [u32::MAX, 0, 0, 0, 0, 0, 0, 0],
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        for instruction in &instructions {
            if instruction.words[0] != u32::MAX {
                validate_instruction(instruction)?;
            }
        }
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
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx)?,
            memory_config: memory_config.clone(),
            address_space_height: dimensions.addr_space_height,
            cell_pointer_max_bits: dimensions.cell_pointer_max_bits,
            timestamp_max_bits: dimensions.timestamp_max_bits,
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

    /// Uploads expanded history produced by serial interpreter preflight.
    ///
    /// The interpreter already records full block values. Chronology still
    /// runs once on the GPU to derive predecessor indexes and touched blocks
    /// in the same format as compiled preflight.
    pub(crate) fn upload_history(
        &self,
        history: &PreflightHistory,
        boundary: ConnectorBoundary,
        initial_memory_images: &[DeviceBufferView],
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let write_masks = validated_history_write_masks(history, &self.memory_config)?;
        let program_log = upload(&history.program, &self.device_ctx)?;
        let memory_log = upload(&history.memory.accesses, &self.device_ctx)?;
        let field_values = upload(&history.memory.field_values, &self.device_ctx)?;
        let write_masks = upload(&write_masks, &self.device_ctx)?;
        let error = [0u32].to_device_on(&self.device_ctx)?;
        self.index_device_history(
            GpuUnindexedHistory::new(program_log, memory_log, field_values, write_masks, error)?,
            initial_memory_images,
            boundary,
        )
    }

    /// Finalizes device-resident program and memory logs into the immutable
    /// transcript and replay plan shared by every trace generator.
    ///
    /// Producers may build the logs differently, but chronology and program
    /// indexing deliberately have one owner and one validation path.
    pub(crate) fn index_device_history(
        &self,
        history: GpuUnindexedHistory,
        initial_memory_images: &[DeviceBufferView],
        boundary: ConnectorBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let GpuUnindexedHistory {
            program: program_log,
            memory: memory_log,
            field_values,
            write_masks,
            error,
        } = history;
        let (initial_write_log, field_initial_values, memory_index) =
            info_span!("postflight_memory_chronology").in_scope(|| {
                build_gpu_memory_chronology(GpuMemoryChronologyInput {
                    memory: &memory_log,
                    write_masks: &write_masks,
                    field_values: &field_values,
                    initial_memory: initial_memory_images,
                    address_space_height: self.address_space_height,
                    pointer_max_bits: self.cell_pointer_max_bits(),
                    address_spaces: self.memory_address_spaces.view(),
                    error: &error,
                    device_ctx: self.device_ctx(),
                })
            })?;
        drop(write_masks);

        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log,
            memory_log,
            initial_write_log,
            field_values,
            field_initial_values,
            memory_predecessors: memory_index.predecessors,
            touched_blocks: memory_index.touched_blocks,
            num_touched_blocks: memory_index.num_touched_blocks,
            error,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = info_span!("postflight_program_index").in_scope(|| {
            GpuPostflightPlan::build(
                self,
                &transcript,
                boundary,
                self.identity.clone(),
                segment_identity,
            )
        })?;
        Ok((transcript, plan))
    }

    /// CUDA device and stream that own this uploaded program.
    pub const fn device_ctx(&self) -> &GpuDeviceCtx {
        &self.device_ctx
    }

    /// Validates the system-owned buffers before postflight borrows any views
    /// or launches work that dereferences them.
    pub(crate) fn validate_system_inputs(
        &self,
        program_device_ctx: &GpuDeviceCtx,
        memory_device_ctx: &GpuDeviceCtx,
        initial_memory: &[Arc<DeviceBuffer<u8>>],
    ) -> Result<(), GpuPostflightError> {
        ensure_same_context(&self.device_ctx, program_device_ctx)?;
        ensure_same_context(&self.device_ctx, memory_device_ctx)?;
        let byte_lengths = initial_memory
            .iter()
            .map(|image| image.len())
            .collect::<Vec<_>>();
        validate_initial_memory_lengths(&self.memory_config, &byte_lengths)
    }

    /// Configured timestamp width validated when this program was uploaded.
    pub const fn timestamp_max_bits(&self) -> u32 {
        self.timestamp_max_bits
    }

    /// Pointer width for address-space-native memory cells.
    pub const fn cell_pointer_max_bits(&self) -> u32 {
        self.cell_pointer_max_bits
    }
}

impl GpuPostflightProgram {
    pub fn ensure_replay_inputs(
        &self,
        transcript: &GpuPostflightTranscript,
        plan: &GpuPostflightPlan,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), GpuPostflightError> {
        ensure_same_context(&self.device_ctx, device_ctx)?;
        ensure_same_context(&transcript.device_ctx, device_ctx)?;
        ensure_same_context(&plan.device_ctx, device_ctx)?;
        if !Arc::ptr_eq(&self.identity, &transcript.program_identity)
            || !Arc::ptr_eq(&self.identity, &plan.program_identity)
        {
            return Err(GpuPostflightError::ProgramMismatch);
        }
        if !Arc::ptr_eq(&transcript.segment_identity, &plan.segment_identity) {
            return Err(GpuPostflightError::SegmentMismatch);
        }
        Ok(())
    }
}

fn ensure_same_context(
    expected: &GpuDeviceCtx,
    actual: &GpuDeviceCtx,
) -> Result<(), GpuPostflightError> {
    if expected.device_id == actual.device_id && expected.stream == actual.stream {
        Ok(())
    } else {
        Err(GpuPostflightError::ContextMismatch)
    }
}

fn instruction_to_replay<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Result<GpuReplayInstruction, GpuPostflightError> {
    let opcode = u32::try_from(instruction.opcode.as_usize())
        .map_err(|_| GpuPostflightError::OpcodeTooLarge(instruction.opcode.as_usize()))?;
    Ok(GpuReplayInstruction {
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

pub(crate) fn gpu_buffer<T>(len: usize, device_ctx: &GpuDeviceCtx) -> DeviceBuffer<T> {
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

struct GpuChronologyCounts {
    num_seeds: usize,
    num_touched: usize,
    non_register_begin: u32,
    field_begin: u32,
    field_end: u32,
    field_seed_base: u32,
    num_field_seeds: usize,
}

impl GpuChronologyCounts {
    fn validated(
        counts: &[u32],
        num_entries: usize,
        num_field_values: usize,
    ) -> Result<Self, GpuPostflightError> {
        let [num_seeds, num_touched, non_register_begin, field_counts @ ..] = counts else {
            return Err(GpuPostflightError::InvalidTranscript(
                "GPU memory chronology returned an invalid count shape".to_string(),
            ));
        };
        let (field_begin, field_end, field_seed_base, num_field_seeds) = match field_counts {
            [] if num_field_values == 0 => (0, 0, 0, 0),
            [field_begin, field_end, field_seed_base, num_field_seeds] if num_field_values != 0 => {
                (*field_begin, *field_end, *field_seed_base, *num_field_seeds)
            }
            _ => {
                return Err(GpuPostflightError::InvalidTranscript(
                    "GPU memory chronology returned an invalid field-count shape".to_string(),
                ));
            }
        };
        let counts = Self {
            num_seeds: *num_seeds as usize,
            num_touched: *num_touched as usize,
            non_register_begin: *non_register_begin,
            field_begin,
            field_end,
            field_seed_base,
            num_field_seeds: num_field_seeds as usize,
        };
        if counts.num_seeds > num_entries
            || counts.num_touched > num_entries
            || counts.non_register_begin as usize > num_entries
            || counts.field_end < counts.field_begin
            || counts.field_end as usize > num_entries
            || (counts.field_end - counts.field_begin) as usize != num_field_values
            || counts.field_seed_base as usize > counts.num_seeds
            || counts.num_field_seeds > counts.num_seeds - counts.field_seed_base as usize
        {
            return Err(GpuPostflightError::InvalidTranscript(
                "GPU memory chronology produced invalid counts".to_string(),
            ));
        }
        Ok(counts)
    }
}

struct GpuMemoryChronologyInput<'a> {
    memory: &'a DeviceBuffer<PreflightMemoryEvent>,
    write_masks: &'a DeviceBuffer<u8>,
    field_values: &'a DeviceBuffer<PreflightFieldBlock>,
    initial_memory: &'a [DeviceBufferView],
    address_space_height: u32,
    pointer_max_bits: u32,
    address_spaces: DeviceBufferView,
    error: &'a DeviceBuffer<u32>,
    device_ctx: &'a GpuDeviceCtx,
}

fn build_gpu_memory_chronology(
    input: GpuMemoryChronologyInput<'_>,
) -> Result<
    (
        DeviceBuffer<PreflightInitialWrite>,
        DeviceBuffer<PreflightFieldBlock>,
        GpuMemoryIndex,
    ),
    GpuPostflightError,
> {
    let GpuMemoryChronologyInput {
        memory,
        write_masks,
        field_values,
        initial_memory,
        address_space_height,
        pointer_max_bits,
        address_spaces,
        error,
        device_ctx,
    } = input;
    // Every history producer assigns the k-th FIELD32 event in memory-log order
    // reference k and allocates exactly one sidecar entry per such event.
    // That dense unique mapping is the race-freedom invariant for in-place GPU
    // resolution; chronology range-checks references but deliberately does not
    // allocate a claimed bitmap or perform another full-log scan to re-prove it.
    if memory.len() != write_masks.len() {
        return Err(GpuPostflightError::InvalidTranscript(
            "memory event and write-mask lengths differ".to_string(),
        ));
    }
    if memory.is_empty() {
        if !field_values.is_empty() {
            return Err(GpuPostflightError::InvalidTranscript(
                "field values exist without memory events".to_string(),
            ));
        }
        return Ok((
            DeviceBuffer::new(),
            DeviceBuffer::new(),
            GpuMemoryIndex {
                predecessors: DeviceBuffer::new(),
                touched_blocks: DeviceBuffer::new(),
                num_touched_blocks: 0,
            },
        ));
    }

    let num_entries = memory.len();
    // Chronology count/sort uses one u64 per event. Value resolution reuses
    // the same allocation as one 16-byte scan element per event.
    let workspace = gpu_buffer::<u64>(2 * num_entries, device_ctx);
    let sorted_keys = gpu_buffer::<u64>(num_entries, device_ctx);
    let predecessors = gpu_buffer::<u32>(num_entries, device_ctx);
    let initial_memory = upload(initial_memory, device_ctx)?;
    // Field metadata extends this tiny counter buffer only when a field
    // sidecar actually exists.
    let count_len = if field_values.is_empty() { 3 } else { 7 };
    let device_counts = upload(&vec![0u32; count_len], device_ctx)?;
    let mut temp_bytes = 0usize;
    unsafe {
        postflight::memory_chronology_get_temp_bytes(
            num_entries,
            &mut temp_bytes,
            device_ctx.stream.as_raw(),
        )?;
    }
    let temp_storage = gpu_buffer::<u8>(temp_bytes, device_ctx);
    unsafe {
        postflight::memory_chronology_sort_and_count(
            memory.view(),
            write_masks.view(),
            field_values.view(),
            address_spaces,
            ADDR_SPACE_OFFSET,
            address_space_height,
            pointer_max_bits,
            DEFERRAL_AS,
            !field_values.is_empty(),
            &workspace,
            &sorted_keys,
            &device_counts,
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    let counts = device_counts.to_host_on(device_ctx)?;
    drop(device_counts);
    let sort_error = error.to_host_on(device_ctx)?[0];
    if sort_error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "GPU memory chronology rejected history with code {sort_error}"
        )));
    }
    let counts = GpuChronologyCounts::validated(&counts, num_entries, field_values.len())?;
    let seeds = gpu_buffer::<PreflightInitialWrite>(counts.num_seeds, device_ctx);
    let field_seeds = gpu_buffer::<PreflightFieldBlock>(counts.num_field_seeds, device_ctx);
    let touched_blocks = gpu_buffer::<TouchedBlock<BabyBear>>(counts.num_touched, device_ctx);
    unsafe {
        postflight::memory_chronology_resolve(
            memory.view(),
            write_masks.view(),
            address_spaces,
            initial_memory.view(),
            field_values.view(),
            ADDR_SPACE_OFFSET,
            counts.non_register_begin,
            &sorted_keys,
            &workspace,
            &predecessors,
            seeds.view(),
            field_seeds.view(),
            counts.field_begin,
            counts.field_end,
            counts.field_seed_base,
            touched_blocks.view(),
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    let resolve_error = error.to_host_on(device_ctx)?[0];
    if resolve_error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "GPU memory chronology failed with code {resolve_error}"
        )));
    }
    // The error copy fences chronology. Release every sort/scan-only
    // allocation before trace generation receives the retained logs and
    // predecessor index.
    drop((workspace, sorted_keys, initial_memory, temp_storage));
    Ok((
        seeds,
        field_seeds,
        GpuMemoryIndex {
            predecessors,
            touched_blocks,
            num_touched_blocks: counts.num_touched,
        },
    ))
}

/// Device-resident transcript and its one generic predecessor index.
///
/// This object is shared across all opcode kernels for the segment. `error` is
/// also shared so replay validation can fail closed without a per-chip copy.
/// Read it once after all replay kernels. A nonzero result is terminal for that
/// proving attempt because threads from other rows may already have updated
/// shared lookup histograms.
pub struct GpuPostflightTranscript {
    program_log: DeviceBuffer<PreflightProgramEvent>,
    memory_log: DeviceBuffer<PreflightMemoryEvent>,
    initial_write_log: DeviceBuffer<PreflightInitialWrite>,
    field_values: DeviceBuffer<PreflightFieldBlock>,
    field_initial_values: DeviceBuffer<PreflightFieldBlock>,
    memory_predecessors: DeviceBuffer<u32>,
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
    error: DeviceBuffer<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuPostflightTranscript {
    /// Reads the shared replay error after synchronizing prior work on its stream.
    pub fn error_code(&self) -> Result<u32, MemCopyError> {
        Ok(self.error.to_host_on(&self.device_ctx)?[0])
    }

    /// Reads the shared replay error, falling back to an explicit fence if the copy fails.
    pub(crate) fn finish_replay(&self) -> Result<u32, GpuPostflightError> {
        match self.error_code() {
            Ok(error) => Ok(error),
            Err(error) => {
                self.synchronize()?;
                Err(error.into())
            }
        }
    }

    #[cfg(feature = "perf-metrics")]
    pub(crate) fn copy_program_log(&self) -> Result<Vec<PreflightProgramEvent>, MemCopyError> {
        self.program_log.to_host_on(&self.device_ctx)
    }

    /// Waits for every replay kernel submitted on this transcript's stream.
    ///
    /// Safe orchestration code must call this before returning ownership to a
    /// caller that may release the transcript or replay plan. This fence is
    /// separate from [`Self::error_code`] so a failed D2H enqueue cannot skip
    /// synchronization of earlier kernels.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.device_ctx.stream.synchronize()
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

    /// Full-width raw-Montgomery values for field-cell memory events.
    pub fn field_values(&self) -> DeviceBufferView {
        self.field_values.view()
    }

    /// Full-width raw-Montgomery values for first-write field seeds.
    pub fn field_initial_values(&self) -> DeviceBufferView {
        self.field_initial_values.view()
    }

    pub fn memory_predecessors(&self) -> DeviceBufferView {
        self.memory_predecessors.view()
    }

    /// Sorted unique final state of every block touched by a timed memory
    /// event, together with the initialized prefix length.
    pub(crate) fn touched_blocks_on(
        &self,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(&DeviceBuffer<TouchedBlock<BabyBear>>, usize), GpuPostflightError> {
        ensure_same_context(&self.device_ctx, device_ctx)?;
        debug_assert!(self.num_touched_blocks <= self.touched_blocks.len());
        Ok((&self.touched_blocks, self.num_touched_blocks))
    }

    pub fn error_ptr(&self) -> *mut u32 {
        self.error.as_mut_ptr()
    }
}

/// The opcode-partitioned replay work list, uploaded once per segment.
pub struct GpuPostflightPlan {
    steps: DeviceBuffer<GpuReplayStep>,
    program_frequencies: DeviceBuffer<u32>,
    opcode_ranges: BTreeMap<u32, Range<usize>>,
    from_state: ExecutionState<u32>,
    to_state: ExecutionState<u32>,
    exit_code: Option<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

fn validated_opcode_ranges(
    active_opcodes: &[u32],
    ranges: &[u32],
    num_steps: usize,
) -> Result<BTreeMap<u32, Range<usize>>, GpuPostflightError> {
    let expected_range_values = active_opcodes.len().checked_mul(2).ok_or_else(|| {
        GpuPostflightError::InvalidTranscript("GPU opcode range count overflow".to_string())
    })?;
    if ranges.len() != expected_range_values {
        return Err(GpuPostflightError::InvalidTranscript(
            "GPU opcode range count does not match active opcodes".to_string(),
        ));
    }
    let mut opcode_ranges = BTreeMap::new();
    let mut covered = 0usize;
    for (&opcode, range) in active_opcodes.iter().zip(ranges.chunks_exact(2)) {
        let start = range[0] as usize;
        let end = range[1] as usize;
        if start != covered || start > end || end > num_steps {
            return Err(GpuPostflightError::InvalidTranscript(
                "GPU opcode ranges do not form a complete partition".to_string(),
            ));
        }
        if start != end {
            opcode_ranges.insert(opcode, start..end);
        }
        covered = end;
    }
    if covered != num_steps {
        return Err(GpuPostflightError::InvalidTranscript(
            "GPU opcode ranges do not cover every execution step".to_string(),
        ));
    }
    Ok(opcode_ranges)
}

impl GpuPostflightPlan {
    fn build(
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        boundary: ConnectorBoundary,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuPostflightError> {
        let num_program_events = transcript.program_log.len();
        if num_program_events == 0 {
            return Err(GpuPostflightError::InvalidTranscript(
                "transcript must contain a final sentinel".to_string(),
            ));
        }
        let num_steps = num_program_events - 1;
        if num_steps >= u32::MAX as usize {
            return Err(GpuPostflightError::InvalidTranscript(
                "program log has more than u32::MAX entries".to_string(),
            ));
        }
        let opcode_keys_in = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let opcode_keys_out = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let steps_in = gpu_buffer::<GpuReplayStep>(num_steps, &program.device_ctx);
        let steps_out = gpu_buffer::<GpuReplayStep>(num_steps, &program.device_ctx);
        let ranges = gpu_buffer::<u32>(2 * program.active_opcodes.len(), &program.device_ctx);
        let program_frequencies = gpu_buffer::<u32>(program.num_program_rows, &program.device_ctx);
        program_frequencies.fill_zero_on(&program.device_ctx)?;
        let mut temp_bytes = 0usize;
        unsafe {
            postflight::program_index_get_temp_bytes(
                num_steps,
                &mut temp_bytes,
                program.device_ctx.stream.as_raw(),
            )?;
        }
        let temp_storage = gpu_buffer::<u8>(temp_bytes, &program.device_ctx);
        let (endpoint_kind, resume_pc, final_timestamp) = if boundary.2.is_some() {
            (0, 0, 0)
        } else {
            (1, boundary.1.pc, boundary.1.timestamp)
        };
        unsafe {
            postflight::program_index(
                program.instructions.view(),
                program.dense_program_rows.view(),
                program.pc_base,
                transcript.program_log.view(),
                transcript.memory_log.view(),
                program.d_active_opcodes.view(),
                program.timestamp_max_bits(),
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
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "GPU postflight rejected transcript with code {error}"
            )));
        }
        let opcode_ranges = validated_opcode_ranges(&program.active_opcodes, &ranges, num_steps)?;
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
    pub(crate) fn program_frequencies_on(
        &self,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<&DeviceBuffer<u32>, GpuPostflightError> {
        ensure_same_context(&self.device_ctx, device_ctx)?;
        Ok(&self.program_frequencies)
    }

    /// Connector inputs derived from the same host events uploaded into this
    /// validated replay plan. This metadata is cold and adds nothing to the
    /// preflight hot-path logs.
    pub(crate) const fn connector_boundary(&self) -> ConnectorBoundary {
        (self.from_state, self.to_state, self.exit_code)
    }

    pub fn opcode_range(&self, opcode: VmOpcode) -> Range<usize> {
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
}
