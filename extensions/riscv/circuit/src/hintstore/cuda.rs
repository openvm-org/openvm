use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64HintStoreOpcode::{HINT_BUFFER, HINT_STORED};
use openvm_stark_backend::prover::AirProvingContext;

use crate::Rv64HintStoreCols;

#[derive(new)]
pub struct Rv64HintStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Rv64HintStoreChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let stored_range = replay_plan.opcode_range(HINT_STORED.global_opcode());
        let buffer_range = replay_plan.opcode_range(HINT_BUFFER.global_opcode());
        let step_range = if stored_range.is_empty() {
            buffer_range
        } else if buffer_range.is_empty() {
            stored_range
        } else {
            if stored_range.end != buffer_range.start {
                return Err(GpuPostflightError::InvalidTranscript(
                    "hint-store opcode ranges are not contiguous".to_string(),
                ));
            }
            stored_range.start..buffer_range.end
        };
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let num_steps = step_range.len();
        let d_counts = openvm_cuda_common::d_buffer::DeviceBuffer::<u32>::with_capacity_on(
            num_steps, device_ctx,
        );
        let opcodes = [
            HINT_STORED.global_opcode().as_usize() as u32,
            HINT_BUFFER.global_opcode().as_usize() as u32,
        ];
        unsafe {
            crate::cuda_abi::hintstore_cuda::replay_count(
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                num_steps,
                opcodes,
                [RV64_REGISTER_AS, RV64_MEMORY_AS],
                self.pointer_max_bits as u32,
                &d_counts,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        let counts = d_counts.to_host_on(device_ctx)?;
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "hint-store replay validation failed with code {error}"
            )));
        }
        drop(d_counts);

        let mut row_offsets = Vec::with_capacity(num_steps + 1);
        row_offsets.push(0u32);
        for count in counts {
            let next = row_offsets
                .last()
                .unwrap()
                .checked_add(count)
                .ok_or_else(|| {
                    GpuPostflightError::InvalidTranscript(
                        "hint-store replay row count exceeds u32".to_string(),
                    )
                })?;
            row_offsets.push(next);
        }
        let rows_used = *row_offsets.last().unwrap() as usize;
        let d_row_offsets = row_offsets.to_device_on(device_ctx)?;
        let trace_height = next_power_of_two_or_zero(rows_used);
        let width = Rv64HintStoreCols::<u8>::width();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, width, device_ctx);
        unsafe {
            crate::cuda_abi::hintstore_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                num_steps,
                d_row_offsets.view(),
                opcodes,
                [RV64_REGISTER_AS, RV64_MEMORY_AS],
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
