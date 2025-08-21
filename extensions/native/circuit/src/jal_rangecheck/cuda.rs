use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{JalRangeCheckCols, JalRangeCheckRecord};
use crate::cuda_abi::native_jal_rangecheck_cuda;

#[derive(new)]
pub struct JalRangeCheckGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for JalRangeCheckGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<JalRangeCheckRecord<F>>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let width = JalRangeCheckCols::<F>::width();

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_jal_rangecheck_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                width as u32,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
