use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::Rv64AuipcOpcode,
};

use crate::{
    adapters::{Rv64RdWriteAdapterCols, Rv64RdWriteAdapterRecord},
    cuda_abi::auipc_cuda,
    Rv64AuipcCoreCols, Rv64AuipcCoreRecord,
};

#[derive(new)]
pub struct Rv64AuipcChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64AuipcChipGpu {
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(Rv64AuipcOpcode::AUIPC.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64AuipcCoreCols::<F>::width() + Rv64RdWriteAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(step_range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            auipc_cuda::replay_tracegen(
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
                Rv64AuipcOpcode::AUIPC.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64AuipcChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Rv64RdWriteAdapterRecord, Rv64AuipcCoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv64AuipcCoreCols::<F>::width() + Rv64RdWriteAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            auipc_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
