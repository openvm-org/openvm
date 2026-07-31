use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        BLOCK_FE_WIDTH,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{ShiftImmOpcode, ShiftWImmOpcode};
use openvm_stark_backend::prover::AirProvingContext;

use super::ShiftRightArithmeticImmCoreCols;
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterCols, Rv64BaseAluWImmU16AdapterCols, RV64_WORD_U16_LIMBS, U16_BITS,
    },
    cuda_abi::{shift_right_arithmetic_imm_cuda, shift_w_right_arithmetic_imm_cuda},
};

#[derive(new)]
pub struct Rv64ShiftRightArithmeticImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Rv64ShiftRightArithmeticImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = replay_plan.opcode_range(ShiftImmOpcode::SRAI.global_opcode());
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluImmU16AdapterCols::<F>::width()
            + ShiftRightArithmeticImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            shift_right_arithmetic_imm_cuda::replay_tracegen(
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
                ShiftImmOpcode::SRAI.global_opcode().as_usize() as u32,
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

#[derive(new)]
pub struct Rv64ShiftWRightArithmeticImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Rv64ShiftWRightArithmeticImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = replay_plan.opcode_range(ShiftWImmOpcode::SRAIW.global_opcode());
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWImmU16AdapterCols::<F>::width()
            + ShiftRightArithmeticImmCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            shift_w_right_arithmetic_imm_cuda::replay_tracegen(
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
                ShiftWImmOpcode::SRAIW.global_opcode().as_usize() as u32,
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
