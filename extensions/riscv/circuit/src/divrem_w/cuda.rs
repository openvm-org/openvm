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

use crate::{
    adapters::{Rv64MultWAdapterCols, Rv64MultWAdapterRecord},
    cuda_abi::{divrem_w_cuda::tracegen, UInt2},
    DivRemCoreCols, DivRemCoreRecord,
};

#[derive(new)]
pub struct Rv64DivRemWChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
    /// M-GPUDEC shared decode state (device operand table + per-segment
    /// emission mode).
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64DivRemWChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64MultWAdapterRecord,
            DivRemCoreRecord<RV64_WORD_NUM_LIMBS>,
        )>();
        #[cfg(feature = "rvr")]
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        #[cfg(feature = "rvr")]
        let delta_records = self.rvr_decode.device_delta_records(
            crate::rvr_gpu_decode::DeltaAirKind::DivRemW,
            &self.range_checker.device_ctx,
        );
        #[cfg(feature = "rvr")]
        let no_delta_records = delta_records.is_none();
        #[cfg(not(feature = "rvr"))]
        let no_delta_records = true;
        if records.is_empty() && no_delta_records {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = DivRemCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>::width()
            + Rv64MultWAdapterCols::<F>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);
        let device_ctx = &self.range_checker.device_ctx;
        // M-GPUDEC (G2): this segment's arena carries compact wire records —
        // decode them on device against the per-exe operand table.
        #[cfg(feature = "rvr")]
        if rvr_wire || delta_records.is_some() {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            assert_eq!(
                delta_records
                    .as_ref()
                    .map_or(records.len(), |buf| buf.len())
                    % PREFLIGHT_ADDSUB_RECORD_SIZE,
                0,
                "compact arena stride mismatch"
            );
            let compact_len = delta_records
                .as_ref()
                .map_or(records.len(), |buf| buf.len());
            let trace_height =
                next_power_of_two_or_zero(compact_len / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (d_table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact segment without a bound operand table");
            let d_records = delta_records
                .unwrap_or_else(|| Arc::new(records.to_device_on(device_ctx).unwrap()));
            let d_trace =
                DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
            unsafe {
                crate::cuda_abi::divrem_w_cuda::tracegen_compact(
                    d_trace.buffer(),
                    trace_height,
                    trace_width,
                    &d_records,
                    &d_table,
                    pc_base,
                    &self.range_checker.count,
                    &self.bitwise_lookup.count,
                    &self.range_tuple_checker.count,
                    tuple_checker_sizes,
                    self.timestamp_max_bits as u32,
                    device_ctx.stream.as_raw(),
                )
                .unwrap();
            }
            return AirProvingContext::simple_no_pis(d_trace);
        }

        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(padded_height, trace_width, device_ctx);
        unsafe {
            tracegen(
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
