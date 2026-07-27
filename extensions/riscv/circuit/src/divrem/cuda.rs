use std::sync::Arc;

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_instructions::riscv::{RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::DivRemOpcode,
};

use crate::{
    adapters::Rv64MultAdapterCols,
    cuda_abi::{divrem_cuda, UInt2},
    DivRemCoreCols,
};

#[derive(new)]
pub struct Rv64DivRemChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64DivRemChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let div = replay_plan.opcode_range(DivRemOpcode::DIV.global_opcode());
        let divu = replay_plan.opcode_range(DivRemOpcode::DIVU.global_opcode());
        let rem = replay_plan.opcode_range(DivRemOpcode::REM.global_opcode());
        let remu = replay_plan.opcode_range(DivRemOpcode::REMU.global_opcode());
        let rows = [div.len(), divu.len(), rem.len(), remu.len()]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript("div/rem replay row count overflow".into())
            })?;
        if rows == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = DivRemCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultAdapterCols::<F>::width();
        let height = next_power_of_two_or_zero(rows);
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let sizes = self.range_tuple_checker.sizes;
        unsafe {
            divrem_cuda::replay_tracegen(
                trace.buffer(),
                height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                div.start,
                div.len(),
                divu.start,
                divu.len(),
                rem.start,
                rem.len(),
                remu.start,
                remu.len(),
                transcript.error_ptr(),
                DivRemOpcode::DIV.global_opcode().as_usize() as u32,
                DivRemOpcode::DIVU.global_opcode().as_usize() as u32,
                DivRemOpcode::REM.global_opcode().as_usize() as u32,
                DivRemOpcode::REMU.global_opcode().as_usize() as u32,
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
