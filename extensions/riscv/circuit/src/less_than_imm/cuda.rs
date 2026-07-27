use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::BLOCK_FE_WIDTH, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{
        riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
        LocalOpcode,
    },
    openvm_riscv_transpiler::LessThanImmOpcode,
};

use super::LessThanImmCoreCols;
use crate::{
    adapters::{Rv64BaseAluImmU16AdapterCols, U16_BITS},
    cuda_abi::less_than_imm_cuda,
};

#[derive(new)]
pub struct Rv64LessThanImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64LessThanImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let slti_range = replay_plan.opcode_range(LessThanImmOpcode::SLTI.global_opcode());
        let sltiu_range = replay_plan.opcode_range(LessThanImmOpcode::SLTIU.global_opcode());
        let num_steps = slti_range
            .len()
            .checked_add(sltiu_range.len())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "less-than-immediate replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluImmU16AdapterCols::<F>::width()
            + LessThanImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            less_than_imm_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                slti_range.start,
                slti_range.len(),
                sltiu_range.start,
                sltiu_range.len(),
                transcript.error_ptr(),
                LessThanImmOpcode::SLTI.global_opcode().as_usize() as u32,
                LessThanImmOpcode::SLTIU.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_IMM_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
