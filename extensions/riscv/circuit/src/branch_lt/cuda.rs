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
use openvm_riscv_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{BranchAdapterCols, U16_BITS},
    cuda_abi::branch_lt_cuda,
    BranchLessThanCoreCols,
};

#[derive(new)]
pub struct BranchLessThanChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl BranchLessThanChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let blt_range = replay_plan.opcode_range(BranchLessThanOpcode::BLT.global_opcode());
        let bltu_range = replay_plan.opcode_range(BranchLessThanOpcode::BLTU.global_opcode());
        let bge_range = replay_plan.opcode_range(BranchLessThanOpcode::BGE.global_opcode());
        let bgeu_range = replay_plan.opcode_range(BranchLessThanOpcode::BGEU.global_opcode());
        let num_steps = [
            blt_range.len(),
            bltu_range.len(),
            bge_range.len(),
            bgeu_range.len(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "branch-less-than replay row count overflow".to_string(),
            )
        })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = BranchAdapterCols::<F>::width()
            + BranchLessThanCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            branch_lt_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                blt_range.start,
                blt_range.len(),
                bltu_range.start,
                bltu_range.len(),
                bge_range.start,
                bge_range.len(),
                bgeu_range.start,
                bgeu_range.len(),
                transcript.error_ptr(),
                BranchLessThanOpcode::BLT.global_opcode().as_usize() as u32,
                BranchLessThanOpcode::BLTU.global_opcode().as_usize() as u32,
                BranchLessThanOpcode::BGE.global_opcode().as_usize() as u32,
                BranchLessThanOpcode::BGEU.global_opcode().as_usize() as u32,
                REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
