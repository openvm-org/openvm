use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{
        Rv64MultAdapterCols, Rv64MultAdapterRecord, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    cuda_abi::{mul_cuda::tracegen, UInt2},
    MultiplicationCoreCols, MultiplicationCoreRecord,
};

#[derive(new)]
pub struct Rv64MultiplicationChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
    /// M-GPUDEC shared decode state (device operand table + per-segment
    /// emission mode).
    pub rvr_decode: std::sync::Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64MultiplicationChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64MultAdapterRecord,
            MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
        )>();
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width =
            MultiplicationCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width()
                + Rv64MultAdapterCols::<F>::width();

        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);
        let device_ctx = &self.range_checker.device_ctx;
        // M-GPUDEC (G2): this segment's arena carries compact wire records —
        // decode them on device against the per-exe operand table.
        if rvr_wire {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            assert_eq!(
                records.len() % PREFLIGHT_ADDSUB_RECORD_SIZE,
                0,
                "compact arena stride mismatch"
            );
            let trace_height =
                next_power_of_two_or_zero(records.len() / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (d_table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact segment without a bound operand table");
            let d_records = records.to_device_on(device_ctx).unwrap();
            let d_trace =
                DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
            unsafe {
                crate::cuda_abi::mul_cuda::tracegen_compact(
                    d_trace.buffer(),
                    trace_height,
                    &d_records,
                    &d_table,
                    pc_base,
                    &self.range_checker.count,
                    self.range_checker.count.len(),
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

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
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
