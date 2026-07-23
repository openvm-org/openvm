use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    crate::cuda_abi::shift_w_cuda::{
        replay_tracegen_logical as rv64_shift_w_logical_replay_tracegen,
        replay_tracegen_right_arithmetic as rv64_shift_w_right_arithmetic_replay_tracegen,
    },
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::ShiftWOpcode,
};

use crate::{
    adapters::{
        Rv64BaseAluWRegU16AdapterCols, Rv64BaseAluWRegU16AdapterRecord, RV64_WORD_U16_LIMBS,
        U16_BITS,
    },
    cuda_abi::shift_w_cuda::{
        tracegen_logical as rv64_shift_w_logical_tracegen,
        tracegen_right_arithmetic as rv64_shift_w_right_arithmetic_tracegen,
    },
    ShiftLogicalCoreCols, ShiftLogicalCoreRecord, ShiftRightArithmeticCoreCols,
    ShiftRightArithmeticCoreRecord,
};

#[derive(new)]
pub struct Rv64ShiftWLogicalChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[derive(new)]
pub struct Rv64ShiftWRightArithmeticChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64ShiftWLogicalChipGpu {
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let sllw_range = replay_plan.opcode_range(ShiftWOpcode::SLLW.global_opcode());
        let srlw_range = replay_plan.opcode_range(ShiftWOpcode::SRLW.global_opcode());
        let num_steps = sllw_range
            .len()
            .checked_add(srlw_range.len())
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "word-logical-shift-register replay row count overflow".to_string(),
                )
            })?;
        if num_steps == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftLogicalCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(num_steps);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_logical_replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                sllw_range.start,
                sllw_range.len(),
                srlw_range.start,
                srlw_range.len(),
                transcript.error_ptr(),
                ShiftWOpcode::SLLW.global_opcode().as_usize() as u32,
                ShiftWOpcode::SRLW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

#[cfg(feature = "rvr")]
impl Rv64ShiftWRightArithmeticChipGpu {
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let range = replay_plan.opcode_range(ShiftWOpcode::SRAW.global_opcode());
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftRightArithmeticCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(range.len());
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_right_arithmetic_replay_tracegen(
                d_trace.buffer(),
                trace_height,
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
                ShiftWOpcode::SRAW.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64ShiftWLogicalChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64BaseAluWRegU16AdapterRecord,
            ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftLogicalCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_logical_tracegen(
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

impl Chip<DenseRecordArena, GpuBackend> for Rv64ShiftWRightArithmeticChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64BaseAluWRegU16AdapterRecord,
            ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv64BaseAluWRegU16AdapterCols::<F>::width()
            + ShiftRightArithmeticCoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        unsafe {
            rv64_shift_w_right_arithmetic_tracegen(
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
