use std::sync::Arc;

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode, PUBLIC_VALUES_AS,
    },
    openvm_riscv_transpiler::Rv64LoadStoreOpcode,
};

use super::STORE_WORD_VALUE_CELLS;
use crate::{
    adapters::{Rv64StoreMultiByteAdapterCols, RV64_BYTE_BITS},
    cuda_abi::store_word_cuda,
    store::core::StoreCoreCols,
};

#[derive(new)]
pub struct Rv64StoreWordChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64StoreWordChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(Rv64LoadStoreOpcode::STOREW.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64StoreMultiByteAdapterCols::<F>::width()
            + StoreCoreCols::<F, STORE_WORD_VALUE_CELLS>::width();
        let trace_height = next_power_of_two_or_zero(step_range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            store_word_cuda::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                transcript.error_ptr(),
                Rv64LoadStoreOpcode::STOREW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                PUBLIC_VALUES_AS,
                self.pointer_max_bits,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}
