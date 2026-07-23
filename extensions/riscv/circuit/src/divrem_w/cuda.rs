use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_instructions::riscv::{RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
    },
    openvm_instructions::{riscv::RV64_REGISTER_AS, LocalOpcode},
    openvm_riscv_transpiler::DivRemWOpcode,
};

use crate::{
    adapters::{Rv64MultWAdapterCols, Rv64MultWAdapterRecord},
    cuda_abi::{divrem_w_cuda, UInt2},
    DivRemCoreCols, DivRemCoreRecord,
};

#[derive(new)]
pub struct Rv64DivRemWChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
impl Rv64DivRemWChipGpu {
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let div = replay_plan.opcode_range(DivRemWOpcode::DIVW.global_opcode());
        let divu = replay_plan.opcode_range(DivRemWOpcode::DIVUW.global_opcode());
        let rem = replay_plan.opcode_range(DivRemWOpcode::REMW.global_opcode());
        let remu = replay_plan.opcode_range(DivRemWOpcode::REMUW.global_opcode());
        let rows = [div.len(), divu.len(), rem.len(), remu.len()]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript("div/rem-w replay row count overflow".into())
            })?;
        if rows == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let width = DivRemCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultWAdapterCols::<F>::width();
        let height = next_power_of_two_or_zero(rows);
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let sizes = self.range_tuple_checker.sizes;
        unsafe {
            divrem_w_cuda::replay_tracegen(
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
                DivRemWOpcode::DIVW.global_opcode().as_usize() as u32,
                DivRemWOpcode::DIVUW.global_opcode().as_usize() as u32,
                DivRemWOpcode::REMW.global_opcode().as_usize() as u32,
                DivRemWOpcode::REMUW.global_opcode().as_usize() as u32,
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

impl Chip<DenseRecordArena, GpuBackend> for Rv64DivRemWChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64MultWAdapterRecord,
            DivRemCoreRecord<RV64_WORD_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = DivRemCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultWAdapterCols::<F>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(padded_height, trace_width, device_ctx);
        unsafe {
            divrem_w_cuda::tracegen(
                d_trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
