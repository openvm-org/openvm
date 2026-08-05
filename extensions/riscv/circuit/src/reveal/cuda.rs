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
use openvm_instructions::{riscv::REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::prover::AirProvingContext;

use super::RevealCols;
use crate::cuda_abi::reveal_cuda;

#[derive(new)]
pub struct RevealChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl RevealChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(RevealOpcode::REVEAL.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let height = next_power_of_two_or_zero(step_range.len());
        let width = RevealCols::<F>::width();
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        unsafe {
            reveal_cuda::replay_tracegen(
                trace.buffer(),
                height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                transcript.error_ptr(),
                RevealOpcode::REVEAL.global_opcode().as_usize() as u32,
                REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
