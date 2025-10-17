use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{FriReducedOpeningRecordMut, OVERALL_WIDTH};
use crate::cuda_abi::fri_cuda;

#[derive(new)]
pub struct FriReducedOpeningChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for FriReducedOpeningChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // TODO[arayi]: Temporary hack to get mut access to `records`, should have `self` or `&mut
        // self` as a parameter **SAFETY**: `records` should be non-empty at this point
        let records =
            unsafe { std::slice::from_raw_parts_mut(records.as_ptr() as *mut u8, records.len()) };

        let mut record_info = Vec::<RowInfo>::with_capacity(records.len());
        let mut offset = 0;

        while offset < records.len() {
            let prev_offset = offset;
            let record =
                RecordSeeker::<DenseRecordArena, FriReducedOpeningRecordMut<F>, _>::get_record_at(
                    &mut offset,
                    records,
                );
            for idx in 0..record.header.length + 2 {
                record_info.push(RowInfo::new(prev_offset as u32, idx));
            }
        }
        debug_assert!(offset == records.len());

        let d_records = records.to_device().unwrap();
        let d_record_info = record_info.to_device().unwrap();

        let trace_height = next_power_of_two_or_zero(record_info.len());
        let trace_width = OVERALL_WIDTH;
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            fri_cuda::tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                record_info.len(),
                &d_record_info,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct RowInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}
