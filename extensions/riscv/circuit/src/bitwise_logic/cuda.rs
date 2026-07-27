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
use openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::prover::AirProvingContext;

use super::BitwiseLogicCoreCols;
use crate::{
    adapters::{Rv64BaseAluRegAdapterCols, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS},
    cuda_abi::bitwise_logic_cuda,
};

#[derive(new)]
pub struct Rv64BitwiseLogicChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Rv64BitwiseLogicChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let xor_range = replay_plan.opcode_range(BaseAluOpcode::XOR.global_opcode());
        let or_range = replay_plan.opcode_range(BaseAluOpcode::OR.global_opcode());
        let and_range = replay_plan.opcode_range(BaseAluOpcode::AND.global_opcode());
        let num_steps = xor_range
            .len()
            .checked_add(or_range.len())
            .and_then(|count| count.checked_add(and_range.len()))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "bitwise-register replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width =
            BitwiseLogicCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width()
                + Rv64BaseAluRegAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            bitwise_logic_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                xor_range.start,
                xor_range.len(),
                or_range.start,
                or_range.len(),
                and_range.start,
                and_range.len(),
                transcript.error_ptr(),
                BaseAluOpcode::XOR.global_opcode().as_usize() as u32,
                BaseAluOpcode::OR.global_opcode().as_usize() as u32,
                BaseAluOpcode::AND.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
