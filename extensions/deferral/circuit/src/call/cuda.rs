use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV64_BYTE_BITS;
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
    },
    openvm_deferral_transpiler::DeferralOpcode,
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode, DEFERRAL_AS,
    },
};

use super::{
    DeferralCallAdapterCols, DeferralCallAdapterRecord, DeferralCallCoreCols,
    DeferralCallCoreRecord,
};
use crate::{
    cuda_abi::call,
    poseidon2::{DeferralPoseidon2ProducerBuffer, DeferralPoseidon2SharedBuffer},
};

#[derive(new)]
pub struct DeferralCallChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub address_bits: usize,
    pub timestamp_max_bits: usize,
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub poseidon2: DeferralPoseidon2SharedBuffer,
}

#[cfg(feature = "rvr")]
impl DeferralCallChipGpu {
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
        max_trace_height: usize,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(DeferralOpcode::CALL.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let trace_height = step_range
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "Deferral CALL trace height overflow".to_string(),
                )
            })?;
        if trace_height > max_trace_height {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "Deferral CALL padded trace height {trace_height} exceeds segment limit {max_trace_height}"
            )));
        }
        let trace_width =
            DeferralCallAdapterCols::<F>::width() + DeferralCallCoreCols::<F>::width();
        trace_height
            .checked_mul(trace_width)
            .and_then(|elements| elements.checked_mul(size_of::<F>()))
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "Deferral CALL trace allocation overflow".to_string(),
                )
            })?;
        let poseidon_records = step_range.len().checked_mul(2).ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript(
                "Deferral CALL Poseidon producer allocation overflow".to_string(),
            )
        })?;
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        let poseidon2 = DeferralPoseidon2ProducerBuffer::new(poseidon_records, device_ctx);
        unsafe {
            call::replay_tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.field_values(),
                transcript.field_initial_values(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                DeferralOpcode::CALL.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                DEFERRAL_AS,
                self.address_bits as u32,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                &poseidon2.records,
                &poseidon2.counts,
                &poseidon2.idx,
                poseidon2.records.len() / 16,
                self.address_bits,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        transcript.synchronize()?;
        let replay_error = transcript.error_code()?;
        if replay_error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "Deferral CALL tracegen rejected replay with code {replay_error}"
            )));
        }
        self.poseidon2.push(poseidon2);
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralCallChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        type Record = (DeferralCallAdapterRecord<F>, DeferralCallCoreRecord<F>);
        const RECORD_SIZE: usize = size_of::<Record>();

        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let num_records = records.len() / RECORD_SIZE;
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace_width =
            DeferralCallAdapterCols::<F>::width() + DeferralCallCoreCols::<F>::width();
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        let poseidon2 = DeferralPoseidon2ProducerBuffer::new(num_records * 2, device_ctx);

        unsafe {
            call::tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                &d_records,
                num_records,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                &poseidon2.records,
                &poseidon2.counts,
                &poseidon2.idx,
                // Length in F elements; the CUDA side converts to record count.
                poseidon2.records.len(),
                self.address_bits,
                device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral call trace");
        }
        self.poseidon2.push(poseidon2);

        AirProvingContext::simple_no_pis(trace)
    }
}
