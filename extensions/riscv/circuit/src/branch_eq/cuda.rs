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
use openvm_instructions::{riscv::REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::BranchEqualOpcode;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{adapters::BranchAdapterCols, cuda_abi::beq_cuda, BranchEqualCoreCols};

#[derive(new)]
pub struct BranchEqualChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl BranchEqualChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let beq_range = replay_plan.opcode_range(BranchEqualOpcode::BEQ.global_opcode());
        let bne_range = replay_plan.opcode_range(BranchEqualOpcode::BNE.global_opcode());
        let num_steps = beq_range
            .len()
            .checked_add(bne_range.len())
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "branch-equality replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width =
            BranchAdapterCols::<F>::width() + BranchEqualCoreCols::<F, BLOCK_FE_WIDTH>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            beq_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                beq_range.start,
                beq_range.len(),
                bne_range.start,
                bne_range.len(),
                transcript.error_ptr(),
                BranchEqualOpcode::BEQ.global_opcode().as_usize() as u32,
                BranchEqualOpcode::BNE.global_opcode().as_usize() as u32,
                REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
