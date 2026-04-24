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
    adapters::{Rv64MultWAdapterCols, Rv64MultWAdapterRecord, RV64_CELL_BITS, RV64_WORD_NUM_LIMBS},
    cuda_abi::{mul_w_cuda::tracegen, UInt2},
    MultiplicationCoreCols, MultiplicationCoreRecord,
};

#[derive(new)]
pub struct Rv64MulWChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64MulWChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64MultWAdapterRecord,
            MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = MultiplicationCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>::width()
            + Rv64MultWAdapterCols::<F>::width();

        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.bitwise_lookup.count,
                RV64_CELL_BITS,
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
