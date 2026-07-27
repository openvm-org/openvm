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
    openvm_riscv_transpiler::{ShiftImmOpcode, ShiftWImmOpcode},
};

use super::ShiftLogicalImmCoreCols;
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterCols, Rv64BaseAluWImmU16AdapterCols, RV64_WORD_U16_LIMBS, U16_BITS,
    },
    cuda_abi::{shift_logical_imm_cuda, shift_w_logical_imm_cuda},
};

#[derive(new)]
pub struct Rv64ShiftLogicalImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64ShiftLogicalImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let slli_range = replay_plan.opcode_range(ShiftImmOpcode::SLLI.global_opcode());
        let srli_range = replay_plan.opcode_range(ShiftImmOpcode::SRLI.global_opcode());
        let num_steps = slli_range
            .len()
            .checked_add(srli_range.len())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "logical-shift-immediate replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluImmU16AdapterCols::<F>::width()
            + ShiftLogicalImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            shift_logical_imm_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                slli_range.start,
                slli_range.len(),
                srli_range.start,
                srli_range.len(),
                transcript.error_ptr(),
                ShiftImmOpcode::SLLI.global_opcode().as_usize() as u32,
                ShiftImmOpcode::SRLI.global_opcode().as_usize() as u32,
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
pub struct Rv64ShiftWLogicalImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64ShiftWLogicalImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let slliw_range = replay_plan.opcode_range(ShiftWImmOpcode::SLLIW.global_opcode());
        let srliw_range = replay_plan.opcode_range(ShiftWImmOpcode::SRLIW.global_opcode());
        let num_steps = slliw_range
            .len()
            .checked_add(srliw_range.len())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "word-logical-shift-immediate replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWImmU16AdapterCols::<F>::width()
            + ShiftLogicalImmCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            shift_w_logical_imm_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                slliw_range.start,
                slliw_range.len(),
                srliw_range.start,
                srliw_range.len(),
                transcript.error_ptr(),
                ShiftWImmOpcode::SLLIW.global_opcode().as_usize() as u32,
                ShiftWImmOpcode::SRLIW.global_opcode().as_usize() as u32,
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
