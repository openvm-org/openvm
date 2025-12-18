use std::ops::Deref;

use openvm_cuda_backend::{base::DeviceMatrix, types::F};
use openvm_cuda_common::memory_manager::MemTracker;
use p3_matrix::dense::RowMajorMatrix;

use super::{ExpBitsLenCols, ExpBitsLenCpuTraceGenerator};
use crate::{cuda::to_device_or_nullptr, primitives::cuda_abi::exp_bits_len_tracegen};

#[derive(Debug, Default)]
pub struct ExpBitsLenGpuTraceGenerator(pub ExpBitsLenCpuTraceGenerator);

impl Deref for ExpBitsLenGpuTraceGenerator {
    type Target = ExpBitsLenCpuTraceGenerator;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ExpBitsLenGpuTraceGenerator {
    pub fn generate_trace_row_major(self) -> RowMajorMatrix<F> {
        self.0.generate_trace_row_major()
    }

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_trace_device(self) -> DeviceMatrix<F> {
        let mem = MemTracker::start("tracegen.exp_bits_len");
        let records = self.0.requests.into_inner().unwrap();
        let num_valid_rows = records.last().map(|record| record.end_row()).unwrap_or(0);
        let height = num_valid_rows.next_power_of_two();
        let width = ExpBitsLenCols::<u8>::width();

        let trace = DeviceMatrix::with_capacity(height, width);
        trace.buffer().fill_zero().unwrap();

        let records = to_device_or_nullptr(&records).unwrap();
        unsafe {
            exp_bits_len_tracegen(
                &records,
                records.len(),
                trace.buffer(),
                height,
                num_valid_rows,
            )
            .unwrap();
        }

        mem.emit_metrics();
        trace
    }
}
