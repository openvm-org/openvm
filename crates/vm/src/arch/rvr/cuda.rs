//! GPU-owned copies of the immutable program and append-only preflight logs.

use std::sync::Arc;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::{CudaError, MemCopyError},
    stream::GpuDeviceCtx,
};
use openvm_instructions::{instruction::Instruction, program::Program, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent};
use thiserror::Error;

use super::{
    postflight::{RvrReplayData, RvrReplayStep},
    RvrPreflightEndpoint, RvrPreflightTranscript,
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
    opcodes: Vec<u32>,
    pc_base: u32,
    device_ctx: GpuDeviceCtx,
    identity: Arc<()>,
}

impl GpuRvrProgram {
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuRvrInputError> {
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
        let opcodes = instructions
            .iter()
            .map(|instruction| instruction.words[0])
            .collect();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            opcodes,
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
        let replay = RvrReplayData::build(self.pc_base, &self.opcodes, transcript, endpoint)
            .map_err(|error| GpuRvrInputError::InvalidTranscript(error.to_string()))?;
        let segment_identity = Arc::new(());
        let gpu = GpuRvrTranscript::upload(
            transcript,
            &replay,
            &self.device_ctx,
            self.identity.clone(),
            segment_identity.clone(),
        )?;
        let plan = GpuRvrReplayPlan::upload(
            &replay,
            &self.device_ctx,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((gpu, plan))
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
        replay: &RvrReplayData,
        device_ctx: &GpuDeviceCtx,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuRvrInputError> {
        Ok(Self {
            program_log: upload(&transcript.program_log, device_ctx)?,
            memory_log: upload(&transcript.memory_log, device_ctx)?,
            initial_write_log: upload(&transcript.initial_write_log, device_ctx)?,
            memory_predecessors: upload(replay.memory_predecessors(), device_ctx)?,
            error: [0u32].to_device_on(device_ctx)?,
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
    fn upload(
        replay: &RvrReplayData,
        device_ctx: &GpuDeviceCtx,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuRvrInputError> {
        if !replay
            .opcode_ranges()
            .values()
            .all(|range| range.start <= range.end && range.end <= replay.steps().len())
        {
            return Err(GpuRvrInputError::InvalidTranscript(
                "derived opcode range is outside the replay step buffer".to_string(),
            ));
        }
        Ok(Self {
            steps: upload(replay.steps(), device_ctx)?,
            opcode_ranges: replay.opcode_ranges().clone(),
            device_ctx: device_ctx.clone(),
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
}
