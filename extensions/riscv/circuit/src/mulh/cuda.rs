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
    openvm_riscv_transpiler::MulHOpcode,
};

use crate::{
    adapters::{Rv64MultAdapterCols, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS},
    cuda_abi::{mulh_cuda, UInt2},
    MulHCoreCols,
};

#[derive(new)]
pub struct Rv64MulHChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64MulHChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let mulh = replay_plan.opcode_range(MulHOpcode::MULH.global_opcode());
        let mulhsu = replay_plan.opcode_range(MulHOpcode::MULHSU.global_opcode());
        let mulhu = replay_plan.opcode_range(MulHOpcode::MULHU.global_opcode());
        let rows = mulh
            .len()
            .checked_add(mulhsu.len())
            .and_then(|n| n.checked_add(mulhu.len()))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript("mulh replay row count overflow".into())
            })?;
        if rows == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = MulHCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultAdapterCols::<F>::width();
        let height = next_power_of_two_or_zero(rows);
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let sizes = self.range_tuple_checker.sizes;
        unsafe {
            mulh_cuda::replay_tracegen(
                trace.buffer(),
                height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                mulh.start,
                mulh.len(),
                mulhsu.start,
                mulhsu.len(),
                mulhu.start,
                mulhu.len(),
                transcript.error_ptr(),
                MulHOpcode::MULH.global_opcode().as_usize() as u32,
                MulHOpcode::MULHSU.global_opcode().as_usize() as u32,
                MulHOpcode::MULHU.global_opcode().as_usize() as u32,
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
