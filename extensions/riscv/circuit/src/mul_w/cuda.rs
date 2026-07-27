use std::sync::Arc;

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::MulWOpcode,
};

use crate::{
    adapters::{Rv64MultWAdapterCols, RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS},
    cuda_abi::{mul_w_cuda, UInt2},
    MultiplicationCoreCols,
};

#[derive(new)]
pub struct Rv64MulWChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64MulWChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = replay_plan.opcode_range(MulWOpcode::MULW.global_opcode());
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = MultiplicationCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultWAdapterCols::<F>::width();
        let height = next_power_of_two_or_zero(range.len());
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let sizes = self.range_tuple_checker.sizes;
        unsafe {
            mul_w_cuda::replay_tracegen(
                trace.buffer(),
                height,
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
                MulWOpcode::MULW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                &self.range_tuple_checker.count,
                UInt2::new(sizes[0], sizes[1]),
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
