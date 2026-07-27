use derive_new::new;
use openvm_circuit::{system::phantom::PhantomCols, utils::next_power_of_two_or_zero};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};
#[cfg(feature = "rvr")]
use {
    crate::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{LocalOpcode, SystemOpcode},
};

use crate::cuda_abi::phantom;

#[derive(new)]
pub struct PhantomChipGPU {
    device_ctx: GpuDeviceCtx,
}

impl PhantomChipGPU {
    pub fn trace_width() -> usize {
        PhantomCols::<F>::width()
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        program.ensure_replay_inputs(transcript, replay_plan, &self.device_ctx)?;
        let step_range = replay_plan.opcode_range(SystemOpcode::PHANTOM.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_height = next_power_of_two_or_zero(step_range.len());
        let trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            Self::trace_width(),
            &self.device_ctx,
        );
        unsafe {
            phantom::replay_tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                transcript.error_ptr(),
                SystemOpcode::PHANTOM.global_opcode().as_usize() as u32,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
