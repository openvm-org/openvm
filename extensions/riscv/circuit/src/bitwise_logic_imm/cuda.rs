use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::prover::AirProvingContext;

use super::BitwiseLogicImmCoreCols;
use crate::{
    adapters::{Rv64BaseAluImmAdapterCols, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS},
    cuda_abi::bitwise_logic_imm_cuda,
};

#[derive(new)]
pub struct Rv64BitwiseLogicImmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Rv64BitwiseLogicImmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let xori_range = replay_plan.opcode_range(BaseAluImmOpcode::XORI.global_opcode());
        let ori_range = replay_plan.opcode_range(BaseAluImmOpcode::ORI.global_opcode());
        let andi_range = replay_plan.opcode_range(BaseAluImmOpcode::ANDI.global_opcode());
        let num_steps = xori_range
            .len()
            .checked_add(ori_range.len())
            .and_then(|count| count.checked_add(andi_range.len()))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "bitwise-immediate replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width =
            BitwiseLogicImmCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width()
                + Rv64BaseAluImmAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            bitwise_logic_imm_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                xori_range.start,
                xori_range.len(),
                ori_range.start,
                ori_range.len(),
                andi_range.start,
                andi_range.len(),
                transcript.error_ptr(),
                BaseAluImmOpcode::XORI.global_opcode().as_usize() as u32,
                BaseAluImmOpcode::ORI.global_opcode().as_usize() as u32,
                BaseAluImmOpcode::ANDI.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_IMM_AS,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
