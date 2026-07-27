//! GPU indexing and read-only replay for immutable preflight history.
//!
//! This module uploads static program metadata, builds memory chronology and
//! opcode indexes once per segment, and exposes immutable replay inputs to
//! system and instruction trace generators. Compiled checkpoint expansion is
//! an optional producer in `arch::rvr::cuda`; this layer does not depend on it.

use std::sync::Arc;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::{CudaError, MemCopyError},
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    instruction::Instruction, program::Program, riscv::RV64_REGISTER_AS, LocalOpcode, SystemOpcode,
    VmOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
};
use thiserror::Error;

#[cfg(any(test, feature = "test-utils"))]
use crate::arch::Postflight;
use crate::{
    arch::{
        AddressSpaceHostLayout, ExecutionState, MemoryCellType, MemoryConfig, PreflightHistory,
        ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH,
    },
    cuda_abi::postflight,
    system::TouchedBlock,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PostflightInstruction {
    /// Global opcode followed by the seven canonical instruction operands.
    pub words: [u32; 8],
}

/// Location of one executed instruction and the first timed memory event in
/// its timestamp interval. The final program sentinel has no entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GpuReplayStep {
    pub program_index: u32,
    pub memory_start: u32,
}

const _: () = assert!(size_of::<PostflightInstruction>() == 32);
const _: () = assert!(size_of::<TouchedBlock<BabyBear>>() == size_of::<[u32; 8]>());
const _: () = assert!(RV64_REGISTER_AS == ADDR_SPACE_OFFSET);

pub(crate) type PostflightFieldBlock = PreflightFieldBlock;

const _: () = assert!(size_of::<PostflightFieldBlock>() == 4 * size_of::<u32>());

/// Device-side layout metadata for one configured address space.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuMemoryAddressSpace {
    num_cells: u64,
    cell_kind: u32,
    _padding: u32,
}

const _: () = assert!(size_of::<GpuMemoryAddressSpace>() == 16);

const MEMORY_CELL_UNSUPPORTED: u32 = 0;
const MEMORY_CELL_U16: u32 = 1;
const MEMORY_CELL_FIELD32: u32 = 2;

fn memory_cell_kind(layout: MemoryCellType) -> u32 {
    match layout {
        MemoryCellType::U16 => MEMORY_CELL_U16,
        MemoryCellType::F { size: 4 } => MEMORY_CELL_FIELD32,
        _ => MEMORY_CELL_UNSUPPORTED,
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

pub struct GpuPostflightProgram {
    instructions: DeviceBuffer<PostflightInstruction>,
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

impl GpuPostflightProgram {
    /// Uploads the immutable program metadata used by history-driven
    /// postflight and trace generation.
    #[doc(hidden)]
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        Self::upload_validated(program, memory_config, device_ctx, |_| Ok(()))
    }

    pub(crate) fn upload_validated<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
        mut validate_instruction: impl FnMut(&PostflightInstruction) -> Result<(), GpuPostflightError>,
    ) -> Result<Self, GpuPostflightError> {
        if F::ORDER_U32 != BabyBear::ORDER_U32 || size_of::<F>() != size_of::<BabyBear>() {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "GPU postflight currently requires the BabyBear proof field".to_string(),
            ));
        }
        if memory_config.pointer_max_bits > u32::BITS as usize
            || memory_config.addr_space_height >= u32::BITS as usize
            || memory_config.timestamp_max_bits >= u32::BITS as usize
        {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "address-space height, pointer width, and timestamp width must fit u32".to_string(),
            ));
        }
        let address_space_count = 1usize
            .checked_shl(memory_config.addr_space_height as u32)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        let expected_address_spaces = (ADDR_SPACE_OFFSET as usize)
            .checked_add(address_space_count)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        if memory_config.addr_spaces.len() != expected_address_spaces {
            return Err(GpuPostflightError::InvalidMemoryConfig(format!(
                "expected {expected_address_spaces} address-space layouts, found {}",
                memory_config.addr_spaces.len()
            )));
        }
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
        let block_pointer_bits = memory_config
            .pointer_max_bits
            .checked_sub(BLOCK_FE_WIDTH.ilog2() as usize)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig(
                    "pointer width is smaller than one memory block".to_string(),
                )
            })?;
        let label_bits = memory_config
            .addr_space_height
            .checked_add(block_pointer_bits)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig(
                    "address-space and block-pointer label width overflow".to_string(),
                )
            })?;
        if label_bits > u32::BITS as usize {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "address-space and block-pointer label does not fit u32".to_string(),
            ));
        }
        let instructions = program
            .instructions_and_debug_infos
            .iter()
            .map(|entry| match entry {
                Some((instruction, _)) => instruction_to_replay(instruction),
                None => Ok(PostflightInstruction {
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
            address_space_height: memory_config.addr_space_height as u32,
            cell_pointer_max_bits: memory_config.pointer_max_bits as u32,
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

    /// Uploads an interpreter-produced history for CPU/GPU trace comparison.
    ///
    /// Production interpreter proving derives chronology on the device from
    /// the segment-start memory image. Tests already carry exact first-write
    /// seeds, so they can upload the validated predecessor index directly.
    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn upload_history_for_test<F: PrimeField32>(
        &self,
        program: &Program<F>,
        history: &PreflightHistory,
        exit_code: Option<u32>,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let postflight = Postflight::new(program, history, &self.memory_config, exit_code)
            .map_err(|error| GpuPostflightError::InvalidTranscript(error.to_string()))?;
        self.upload_validated_history_for_test(history, postflight)
    }

    /// Uploads an isolated chip fixture whose final sentinel need not resolve
    /// to another instruction in the fixture program.
    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn upload_isolated_history_for_test<F: PrimeField32>(
        &self,
        program: &Program<F>,
        history: &PreflightHistory,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let postflight = Postflight::new_for_test(program, history, &self.memory_config)
            .map_err(|error| GpuPostflightError::InvalidTranscript(error.to_string()))?;
        self.upload_validated_history_for_test(history, postflight)
    }

    #[cfg(any(test, feature = "test-utils"))]
    fn upload_validated_history_for_test<F: PrimeField32>(
        &self,
        history: &PreflightHistory,
        postflight: Postflight<'_, F>,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let replay_steps = postflight
            .replay_steps_for_test()
            .map(|(program_index, memory_start)| GpuReplayStep {
                program_index,
                memory_start,
            })
            .collect::<Vec<_>>();
        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log: upload(&history.program, &self.device_ctx)?,
            memory_log: upload(&history.memory.accesses, &self.device_ctx)?,
            initial_write_log: upload(&history.memory.initial_writes, &self.device_ctx)?,
            field_values: upload(&history.memory.field_values, &self.device_ctx)?,
            field_initial_values: upload(&history.memory.field_initial_values, &self.device_ctx)?,
            memory_predecessors: upload(
                postflight.memory_predecessors_for_test(),
                &self.device_ctx,
            )?,
            touched_blocks: DeviceBuffer::new(),
            num_touched_blocks: 0,
            error: [0u32].to_device_on(&self.device_ctx)?,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = GpuPostflightPlan {
            steps: upload(&replay_steps, &self.device_ctx)?,
            program_frequencies: upload(postflight.filtered_exec_frequencies(), &self.device_ctx)?,
            opcode_ranges: postflight.opcode_ranges_for_test().clone(),
            from_state: postflight.from_state(),
            to_state: postflight.to_state(),
            exit_code: postflight.exit_code(),
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity,
        };
        Ok((transcript, plan))
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
        if history.program.is_empty() {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight history must contain a final program sentinel".to_string(),
            ));
        }
        if history.memory.accesses.len() >= (1usize << 31) {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight memory log exceeds packed predecessor indexes".to_string(),
            ));
        }

        let mut write_masks = Vec::with_capacity(history.memory.accesses.len());
        let field_values = &history.memory.field_values;
        let mut field_cursor = 0usize;
        for event in &history.memory.accesses {
            let address_space = event.address_space() as usize;
            let layout = self
                .memory_config
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
                    || reference >= field_values.len()
                {
                    return Err(GpuPostflightError::InvalidTranscript(
                        "field memory events must use dense ordered sidecar references".to_string(),
                    ));
                }
                field_cursor += 1;
            }
        }
        if field_cursor != field_values.len() {
            return Err(GpuPostflightError::InvalidTranscript(
                "field sidecar contains unreferenced values".to_string(),
            ));
        }

        let program_log = upload(&history.program, &self.device_ctx)?;
        let memory_log = upload(&history.memory.accesses, &self.device_ctx)?;
        let field_values = upload(field_values, &self.device_ctx)?;
        let write_masks = upload(&write_masks, &self.device_ctx)?;
        let error = [0u32].to_device_on(&self.device_ctx)?;
        self.index_device_history(
            program_log,
            memory_log,
            field_values,
            write_masks,
            error,
            initial_memory_images,
            boundary,
        )
    }

    /// Finalizes device-resident program and memory logs into the immutable
    /// transcript and replay plan shared by every trace generator.
    ///
    /// Producers may build the logs differently, but chronology and program
    /// indexing deliberately have one owner and one validation path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn index_device_history(
        &self,
        program_log: DeviceBuffer<PreflightProgramEvent>,
        memory_log: DeviceBuffer<PreflightMemoryEvent>,
        field_values: DeviceBuffer<PostflightFieldBlock>,
        write_masks: DeviceBuffer<u8>,
        error: DeviceBuffer<u32>,
        initial_memory_images: &[DeviceBufferView],
        boundary: ConnectorBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let (initial_write_log, field_initial_values, memory_index) = build_gpu_memory_chronology(
            &memory_log,
            &write_masks,
            &field_values,
            initial_memory_images,
            self.address_space_height,
            self.cell_pointer_max_bits(),
            self.memory_address_spaces.view(),
            &error,
            self.device_ctx(),
        )?;
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
        let plan = GpuPostflightPlan::build(
            self,
            &transcript,
            boundary,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((transcript, plan))
    }

    pub(crate) const fn device_ctx(&self) -> &GpuDeviceCtx {
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

    pub(crate) const fn timestamp_max_bits(&self) -> u32 {
        self.timestamp_max_bits
    }

    pub(crate) const fn cell_pointer_max_bits(&self) -> u32 {
        self.cell_pointer_max_bits
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_test(
        opcodes: &[u32],
        pc_base: u32,
        timestamp_max_bits: u32,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        let instructions = opcodes
            .iter()
            .map(|&opcode| PostflightInstruction {
                words: [opcode, 0, 0, 0, 0, 0, 0, 0],
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes
            .iter()
            .copied()
            .filter(|&opcode| opcode != u32::MAX)
            .collect::<Vec<_>>();
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
        let memory_config = MemoryConfig::default();
        let memory_address_spaces = memory_config
            .addr_spaces
            .iter()
            .map(|config| GpuMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_kind: memory_cell_kind(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            dense_program_rows: upload(&dense_program_rows, device_ctx)?,
            num_program_rows: next_program_row as usize,
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx)?,
            memory_config,
            address_space_height: MemoryConfig::default().addr_space_height as u32,
            cell_pointer_max_bits: MemoryConfig::default().pointer_max_bits as u32,
            timestamp_max_bits,
            pc_base,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        })
    }

    #[cfg(test)]
    pub(crate) fn index_program_log_for_test(
        &self,
        program_log: &[PreflightProgramEvent],
        boundary: ConnectorBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log: upload(program_log, &self.device_ctx)?,
            memory_log: DeviceBuffer::new(),
            initial_write_log: DeviceBuffer::new(),
            field_values: DeviceBuffer::new(),
            field_initial_values: DeviceBuffer::new(),
            memory_predecessors: DeviceBuffer::new(),
            touched_blocks: DeviceBuffer::new(),
            num_touched_blocks: 0,
            error: [0u32].to_device_on(&self.device_ctx)?,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = GpuPostflightPlan::build(
            self,
            &transcript,
            boundary,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((transcript, plan))
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
) -> Result<PostflightInstruction, GpuPostflightError> {
    let opcode = u32::try_from(instruction.opcode.as_usize())
        .map_err(|_| GpuPostflightError::OpcodeTooLarge(instruction.opcode.as_usize()))?;
    Ok(PostflightInstruction {
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

#[allow(clippy::too_many_arguments)]
fn build_gpu_memory_chronology(
    memory: &DeviceBuffer<PreflightMemoryEvent>,
    write_masks: &DeviceBuffer<u8>,
    field_values: &DeviceBuffer<PostflightFieldBlock>,
    initial_memory: &[DeviceBufferView],
    address_space_height: u32,
    pointer_max_bits: u32,
    address_spaces: DeviceBufferView,
    error: &DeviceBuffer<u32>,
    device_ctx: &GpuDeviceCtx,
) -> Result<
    (
        DeviceBuffer<PreflightInitialWrite>,
        DeviceBuffer<PostflightFieldBlock>,
        GpuMemoryIndex,
    ),
    GpuPostflightError,
> {
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
    // Keep the U16-only allocation exactly as before. Field metadata extends
    // this tiny counter buffer only when a field sidecar actually exists.
    let count_len = if field_values.is_empty() { 2 } else { 6 };
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
    let (num_seeds, num_touched) = (counts[0], counts[1]);
    let (field_begin, field_end, field_seed_base, num_field_seeds) = if let [_, _, field_begin, field_end, field_seed_base, num_field_seeds] =
        counts.as_slice()
    {
        (*field_begin, *field_end, *field_seed_base, *num_field_seeds)
    } else {
        (0, 0, 0, 0)
    };
    let sort_error = error.to_host_on(device_ctx)?[0];
    if sort_error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "GPU memory chronology rejected history with code {sort_error}"
        )));
    }
    let num_seeds = num_seeds as usize;
    let num_touched = num_touched as usize;
    let num_field_seeds = num_field_seeds as usize;
    if num_seeds > num_entries
        || num_touched > num_entries
        || field_end < field_begin
        || field_end as usize > num_entries
        || (field_end - field_begin) as usize != field_values.len()
        || field_seed_base as usize > num_seeds
        || num_field_seeds > num_seeds - field_seed_base as usize
    {
        return Err(GpuPostflightError::InvalidTranscript(
            "GPU memory chronology produced invalid counts".to_string(),
        ));
    }
    let seeds = gpu_buffer::<PreflightInitialWrite>(num_seeds, device_ctx);
    let field_seeds = gpu_buffer::<PostflightFieldBlock>(num_field_seeds, device_ctx);
    let touched_blocks = gpu_buffer::<TouchedBlock<BabyBear>>(num_touched, device_ctx);
    unsafe {
        postflight::memory_chronology_resolve(
            memory.view(),
            write_masks.view(),
            address_spaces,
            initial_memory.view(),
            field_values.view(),
            RV64_REGISTER_AS,
            &sorted_keys,
            &workspace,
            &predecessors,
            seeds.view(),
            field_seeds.view(),
            field_begin,
            field_end,
            field_seed_base,
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
            num_touched_blocks: num_touched,
        },
    ))
}

#[cfg(test)]
pub(crate) type ChronologyOutputForTest = (
    Vec<PreflightMemoryEvent>,
    Vec<PreflightInitialWrite>,
    Vec<PostflightFieldBlock>,
    Vec<PostflightFieldBlock>,
    Vec<u32>,
    Vec<TouchedBlock<BabyBear>>,
);

#[cfg(test)]
pub(crate) fn build_memory_chronology_for_test(
    memory: &[PreflightMemoryEvent],
    write_masks: &[u8],
    field_values: &[PostflightFieldBlock],
    initial_memory: &[Vec<u8>],
    config: &MemoryConfig,
) -> Result<ChronologyOutputForTest, GpuPostflightError> {
    let device_ctx = GpuDeviceCtx::for_current_device()?;
    let memory = upload(memory, &device_ctx)?;
    let write_masks = upload(write_masks, &device_ctx)?;
    let field_values = upload(field_values, &device_ctx)?;
    let initial_memory = initial_memory
        .iter()
        .map(|image| upload(image, &device_ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let initial_memory_views = initial_memory
        .iter()
        .map(|image| image.view())
        .collect::<Vec<_>>();
    let address_spaces = config
        .addr_spaces
        .iter()
        .map(|config| GpuMemoryAddressSpace {
            num_cells: config.num_cells as u64,
            cell_kind: memory_cell_kind(config.layout),
            _padding: 0,
        })
        .collect::<Vec<_>>();
    let address_spaces = upload(&address_spaces, &device_ctx)?;
    let error = [0u32].to_device_on(&device_ctx)?;
    let (seeds, field_seeds, index) = build_gpu_memory_chronology(
        &memory,
        &write_masks,
        &field_values,
        &initial_memory_views,
        config.addr_space_height as u32,
        config.pointer_max_bits as u32,
        address_spaces.view(),
        &error,
        &device_ctx,
    )?;
    let mut touched = index.touched_blocks.to_host_on(&device_ctx)?;
    touched.truncate(index.num_touched_blocks);
    Ok((
        memory.to_host_on(&device_ctx)?,
        seeds.to_host_on(&device_ctx)?,
        field_values.to_host_on(&device_ctx)?,
        field_seeds.to_host_on(&device_ctx)?,
        index.predecessors.to_host_on(&device_ctx)?,
        touched,
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
    field_values: DeviceBuffer<PostflightFieldBlock>,
    field_initial_values: DeviceBuffer<PostflightFieldBlock>,
    memory_predecessors: DeviceBuffer<u32>,
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
    error: DeviceBuffer<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuPostflightTranscript {
    pub fn error_code(&self) -> Result<u32, MemCopyError> {
        Ok(self.error.to_host_on(&self.device_ctx)?[0])
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

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn program_log_host(&self) -> Result<Vec<PreflightProgramEvent>, MemCopyError> {
        self.program_log.to_host_on(&self.device_ctx)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn replace_program_log_for_test(
        &mut self,
        program_log: &[PreflightProgramEvent],
    ) -> Result<(), MemCopyError> {
        assert_eq!(
            program_log.len(),
            self.program_log.len(),
            "a replay plan is valid only for its original program-log length"
        );
        self.program_log = program_log.to_device_on(&self.device_ctx)?;
        Ok(())
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn memory_log_host(&self) -> Result<Vec<PreflightMemoryEvent>, MemCopyError> {
        self.memory_log.to_host_on(&self.device_ctx)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn initial_write_log_host(&self) -> Result<Vec<PreflightInitialWrite>, MemCopyError> {
        self.initial_write_log.to_host_on(&self.device_ctx)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn field_values_host(&self) -> Result<Vec<[u32; BLOCK_FE_WIDTH]>, MemCopyError> {
        Ok(self
            .field_values
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|block| block.values)
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn memory_predecessors_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.memory_predecessors.to_host_on(&self.device_ctx)
    }

    pub fn error_ptr(&self) -> *mut u32 {
        self.error.as_mut_ptr()
    }
}

/// The opcode-partitioned replay work list, uploaded once per segment.
pub struct GpuPostflightPlan {
    steps: DeviceBuffer<GpuReplayStep>,
    program_frequencies: DeviceBuffer<u32>,
    opcode_ranges: std::collections::BTreeMap<u32, std::ops::Range<usize>>,
    from_state: ExecutionState<u32>,
    to_state: ExecutionState<u32>,
    exit_code: Option<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
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
        let mut opcode_ranges = std::collections::BTreeMap::new();
        let mut covered = 0usize;
        for (&opcode, range) in program.active_opcodes.iter().zip(ranges.chunks_exact(2)) {
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

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn steps_host(&self) -> Result<Vec<[u32; 2]>, MemCopyError> {
        Ok(self
            .steps
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|step| [step.program_index, step.memory_start])
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn program_frequencies_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.program_frequencies.to_host_on(&self.device_ctx)
    }
}

#[cfg(test)]
mod validation_tests {
    use openvm_instructions::PUBLIC_VALUES_AS;

    use super::*;

    fn configured_byte_lengths(config: &MemoryConfig) -> Vec<usize> {
        config
            .addr_spaces
            .iter()
            .map(|address_space| address_space.num_cells * address_space.layout.size())
            .collect()
    }

    #[test]
    fn field_cells_are_restricted_to_deferral_address_space() {
        let mut config = MemoryConfig::default();
        assert!(validate_field_address_spaces(&config).is_ok());

        config.addr_spaces[PUBLIC_VALUES_AS as usize].layout = MemoryCellType::field32();
        assert!(matches!(
            validate_field_address_spaces(&config),
            Err(GpuPostflightError::InvalidMemoryConfig(_))
        ));
    }

    #[test]
    fn initial_memory_must_match_every_configured_address_space() {
        let config = MemoryConfig::default();
        let mut byte_lengths = configured_byte_lengths(&config);
        assert!(validate_initial_memory_lengths(&config, &byte_lengths).is_ok());

        byte_lengths.pop();
        assert!(matches!(
            validate_initial_memory_lengths(&config, &byte_lengths),
            Err(GpuPostflightError::InvalidTranscript(_))
        ));

        let mut byte_lengths = configured_byte_lengths(&config);
        byte_lengths[PUBLIC_VALUES_AS as usize] -= 1;
        assert!(matches!(
            validate_initial_memory_lengths(&config, &byte_lengths),
            Err(GpuPostflightError::InvalidTranscript(_))
        ));
    }
}
