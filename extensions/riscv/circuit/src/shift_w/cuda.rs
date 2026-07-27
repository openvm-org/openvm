use std::sync::Arc;

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    crate::cuda_abi::shift_w_cuda::{
        replay_tracegen_logical as rv64_shift_w_logical_replay_tracegen,
        replay_tracegen_right_arithmetic as rv64_shift_w_right_arithmetic_replay_tracegen,
    },
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::ShiftWOpcode,
};

use crate::{
    adapters::{Rv64BaseAluWRegU16AdapterCols, RV64_WORD_U16_LIMBS, U16_BITS},
    ShiftLogicalCoreCols, ShiftRightArithmeticCoreCols,
};

#[derive(new)]
pub struct Rv64ShiftWLogicalChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[derive(new)]
pub struct Rv64ShiftWRightArithmeticChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64ShiftWLogicalChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let sllw_range = replay_plan.opcode_range(ShiftWOpcode::SLLW.global_opcode());
        let srlw_range = replay_plan.opcode_range(ShiftWOpcode::SRLW.global_opcode());
        let num_steps = sllw_range
            .len()
            .checked_add(srlw_range.len())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "word-logical-shift-register replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftLogicalCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_logical_replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                sllw_range.start,
                sllw_range.len(),
                srlw_range.start,
                srlw_range.len(),
                transcript.error_ptr(),
                ShiftWOpcode::SLLW.global_opcode().as_usize() as u32,
                ShiftWOpcode::SRLW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

#[cfg(feature = "rvr")]
impl Rv64ShiftWRightArithmeticChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = replay_plan.opcode_range(ShiftWOpcode::SRAW.global_opcode());
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftRightArithmeticCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_right_arithmetic_replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                range.start,
                range.len(),
                transcript.error_ptr(),
                ShiftWOpcode::SRAW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
