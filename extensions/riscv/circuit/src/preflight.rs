//! RV64 checkpoint expansion for GPU preflight.
//!
//! This module converts compact RVR checkpoints into the same immutable
//! history consumed by the generic CUDA postflight indexer. Chronology,
//! opcode indexing, and trace generation live in `arch::cuda::postflight`.

use openvm_circuit::arch::{
    cuda::postflight::{
        GpuPostflightBoundary, GpuPostflightContext, GpuPostflightError, GpuPostflightPlan,
        GpuPostflightProgram, GpuPostflightTranscript, GpuUnindexedHistory,
    },
    rvr::{bridge::read_registers, PreflightEndpoint, PreflightExecution},
    to_byte_ptr_bits, MemoryConfig, PreflightFieldBlock, POSTFLIGHT_PREDECESSOR_INDEX_LIMIT,
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    program::Program,
    riscv::{IMM_AS, MEMORY_AS, REGISTER_AS},
    DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::{PreflightMemoryEvent, PreflightProgramEvent, RvrCheckpoint};

use crate::{cuda_abi::rvr_checkpoint_replay, Rv64ImPreflightGpuTracegen};

fn upload<T>(
    values: &[T],
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<T>, GpuPostflightError> {
    if values.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        Ok(values.to_device_on(device_ctx)?)
    }
}

fn gpu_buffer<T>(len: usize, device_ctx: &GpuDeviceCtx) -> DeviceBuffer<T> {
    if len == 0 {
        DeviceBuffer::new()
    } else {
        DeviceBuffer::with_capacity_on(len, device_ctx)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PostflightEventCount {
    pub(crate) memory: u32,
    pub(crate) field: u32,
}

const _: () = assert!(size_of::<PostflightEventCount>() == 2 * size_of::<u32>());
const _: () = assert!(std::mem::offset_of!(PostflightEventCount, memory) == 0);
const _: () = assert!(std::mem::offset_of!(PostflightEventCount, field) == 4);

mod schedule;
use schedule::RvrReplayAccessSchedule;
pub use schedule::{PostflightAccessRegistry, PostflightAccessSchedule, PostflightAccessSpan};

pub struct PreflightReplayProgram {
    program: GpuPostflightProgram,
    schedule_dispatch: DeviceBuffer<u32>,
    access_schedules: DeviceBuffer<RvrReplayAccessSchedule>,
    access_spans: DeviceBuffer<PostflightAccessSpan>,
    static_values: DeviceBuffer<u64>,
    byte_pointer_max_bits: u32,
}

struct ValidatedReplayBoundary {
    endpoint_kind: u32,
    replay_value_cursor: u32,
}

struct ReplayEventLayout {
    offsets: Vec<PostflightEventCount>,
    total_memory: u32,
    total_fields: u32,
}

impl ReplayEventLayout {
    fn from_counts(counts: &[PostflightEventCount]) -> Result<Self, GpuPostflightError> {
        let mut total_memory = 0u32;
        let mut total_fields = 0u32;
        let mut offsets = Vec::with_capacity(counts.len());
        for count in counts {
            offsets.push(PostflightEventCount {
                memory: total_memory,
                field: total_fields,
            });
            total_memory = total_memory.checked_add(count.memory).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay memory-event count exceeds u32".to_string(),
                )
            })?;
            total_fields = total_fields.checked_add(count.field).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay field-event count exceeds u32".to_string(),
                )
            })?;
        }
        if total_memory >= POSTFLIGHT_PREDECESSOR_INDEX_LIMIT {
            return Err(GpuPostflightError::InvalidTranscript(
                "checkpoint replay memory log exceeds packed predecessor indexes".to_string(),
            ));
        }
        Ok(Self {
            offsets,
            total_memory,
            total_fields,
        })
    }
}

impl PreflightReplayProgram {
    fn validate_execution(
        &self,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<ValidatedReplayBoundary, GpuPostflightError> {
        if execution.retired != num_insns {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "preflight instret {} does not match segment num_insns {num_insns}",
                execution.retired
            )));
        }
        if execution.from_state.timestamp != 1
            || execution.to_state.byte_pc() != execution.state.pc()
            || execution.to_state.timestamp >= (1u32 << self.program.timestamp_max_bits())
        {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight has an invalid boundary".to_string(),
            ));
        }
        let replay_value_cursor =
            u32::try_from(execution.transcript.replay_values.len()).map_err(|_| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay-value stream exceeds the u32 cursor domain".to_string(),
                )
            })?;
        let endpoint_kind = match execution.endpoint {
            PreflightEndpoint::Terminated => 0,
            PreflightEndpoint::Suspended => 1,
        };
        Ok(ValidatedReplayBoundary {
            endpoint_kind,
            replay_value_cursor,
        })
    }

    /// Uploads one program together with the extension schedules used only by
    /// compiled checkpoint expansion.
    pub fn upload_with_postflight_access_registry<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        registry: &PostflightAccessRegistry,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        registry.validate_disjoint_opcodes(Rv64ImPreflightGpuTracegen::replay_opcodes())?;
        let program = GpuPostflightProgram::upload_with_instruction_validation(
            program,
            memory_config,
            device_ctx,
            |instr| {
                registry.validate_instruction(
                    instr,
                    memory_config.pointer_max_bits,
                    memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells,
                )
            },
        )?;
        Ok(Self {
            program,
            schedule_dispatch: upload(&registry.dispatch, device_ctx)?,
            access_schedules: upload(&registry.schedules, device_ctx)?,
            access_spans: upload(&registry.spans, device_ctx)?,
            static_values: upload(&registry.static_values, device_ctx)?,
            byte_pointer_max_bits: to_byte_ptr_bits(memory_config.pointer_max_bits)
                .min(u32::BITS as usize) as u32,
        })
    }

    /// Generic immutable-program view consumed by postflight indexing and
    /// read-only trace generation.
    pub const fn program(&self) -> &GpuPostflightProgram {
        &self.program
    }
}

impl PreflightReplayProgram {
    /// Expands checkpoint execution into ordinary program and memory logs.
    /// Independent chunks emit byte-masked block intents; one device-wide
    /// chronology pass resolves them against the segment-start memory image.
    pub(crate) fn postflight(
        &self,
        context: GpuPostflightContext<'_>,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let program = self.program();
        let initial_registers = context.memory_image(REGISTER_AS)?;
        let initial_memory = context.memory_image(MEMORY_AS)?;
        let boundary = self.validate_execution(execution, num_insns)?;
        let final_registers = read_registers(&execution.state);
        let mut final_anchor = RvrCheckpoint {
            pc: execution.to_state.byte_pc(),
            timestamp: execution.to_state.timestamp,
            retired: execution.retired,
            replay_value_cursor: boundary.replay_value_cursor,
            regs: [0; 31],
        };
        final_anchor.regs.copy_from_slice(&final_registers[1..]);
        let mut anchors = execution.transcript.checkpoints.clone();
        anchors.push(final_anchor);
        let anchors = upload(&anchors, program.device_ctx())?;
        let replay_values = upload(&execution.transcript.replay_values, program.device_ctx())?;
        let error = [0u32].to_device_on(program.device_ctx())?;
        let event_counts = gpu_buffer::<PostflightEventCount>(anchors.len(), program.device_ctx());
        event_counts.fill_zero_on(program.device_ctx())?;
        let address_spaces = [REGISTER_AS, MEMORY_AS, IMM_AS, DEFERRAL_AS];
        let count_span = tracing::info_span!("postflight_replay_count").entered();
        unsafe {
            rvr_checkpoint_replay::count(
                program.instructions(),
                program.pc_base(),
                initial_registers,
                initial_memory,
                anchors.view(),
                replay_values.view(),
                self.schedule_dispatch.view(),
                self.access_schedules.view(),
                self.access_spans.view(),
                self.static_values.view(),
                address_spaces,
                self.byte_pointer_max_bits,
                program.cell_pointer_max_bits(),
                execution.from_state.byte_pc(),
                execution.from_state.timestamp,
                boundary.endpoint_kind,
                &event_counts,
                &error,
                program.device_ctx().stream.as_raw(),
            )?;
        }
        let count_error = error.to_host_on(program.device_ctx())?[0];
        if count_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU count replay rejected execution with code {count_error}"
            )));
        }
        let counts = event_counts.to_host_on(program.device_ctx())?;
        let layout = ReplayEventLayout::from_counts(&counts)?;
        drop(count_span);
        let program_len = usize::try_from(execution.retired)
            .ok()
            .and_then(|retired| retired.checked_add(1))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay program-log length overflow".to_string(),
                )
            })?;
        let program_log = gpu_buffer::<PreflightProgramEvent>(program_len, program.device_ctx());
        let memory_log =
            gpu_buffer::<PreflightMemoryEvent>(layout.total_memory as usize, program.device_ctx());
        let field_values =
            gpu_buffer::<PreflightFieldBlock>(layout.total_fields as usize, program.device_ctx());
        // One transient byte per event is enough to distinguish reads, full
        // writes, and partial block writes. The chronology pass consumes and
        // releases this before opcode trace generation.
        let write_masks = gpu_buffer::<u8>(layout.total_memory as usize, program.device_ctx());
        let offsets = upload(&layout.offsets, program.device_ctx())?;
        let emit_span = tracing::info_span!("postflight_replay_emit").entered();
        unsafe {
            rvr_checkpoint_replay::emit(
                program.instructions(),
                program.pc_base(),
                initial_registers,
                initial_memory,
                anchors.view(),
                replay_values.view(),
                offsets.view(),
                self.schedule_dispatch.view(),
                self.access_schedules.view(),
                self.access_spans.view(),
                self.static_values.view(),
                address_spaces,
                self.byte_pointer_max_bits,
                program.cell_pointer_max_bits(),
                execution.from_state.byte_pc(),
                execution.from_state.timestamp,
                boundary.endpoint_kind,
                program_log.view(),
                memory_log.view(),
                write_masks.view(),
                field_values.view(),
                &error,
                program.device_ctx().stream.as_raw(),
            )?;
        }
        let emit_error = error.to_host_on(program.device_ctx())?[0];
        if emit_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU emit replay rejected execution with code {emit_error}"
            )));
        }
        drop(emit_span);
        // The host reads above synchronize the emit stream. Release compact
        // replay inputs before chronology allocates its sort/scan scratch.
        drop(anchors);
        drop(replay_values);
        drop(event_counts);
        drop(offsets);
        let history =
            GpuUnindexedHistory::new(program_log, memory_log, field_values, write_masks, error)?;
        context.finalize_device_history(
            history,
            GpuPostflightBoundary::new(
                execution.from_state,
                execution.to_state,
                matches!(execution.endpoint, PreflightEndpoint::Terminated).then_some(0),
            ),
        )
    }
}

#[cfg(any(test, feature = "test-utils"))]
mod testing;
