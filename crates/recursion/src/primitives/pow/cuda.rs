use std::sync::Arc;

use openvm_cuda_backend::{base::DeviceMatrix, prelude::F};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, memory_manager::MemTracker};

use crate::primitives::{
    cuda_abi::pow_checker_tracegen,
    pow::{PowerCheckerCols, PowerCheckerCpuTraceGenerator},
};

#[derive(Debug)]
pub struct PowerCheckerGpuTraceGenerator<const BASE: usize, const N: usize> {
    pow_count: DeviceBuffer<u32>,
    range_count: DeviceBuffer<u32>,
    cpu_checker: Option<Arc<PowerCheckerCpuTraceGenerator<BASE, N>>>,
}

impl<const BASE: usize, const N: usize> PowerCheckerGpuTraceGenerator<BASE, N> {
    pub fn new(cpu_checker: Option<Arc<PowerCheckerCpuTraceGenerator<BASE, N>>>) -> Self {
        let pow_count = DeviceBuffer::with_capacity(N);
        pow_count.fill_zero().unwrap();
        let range_count = DeviceBuffer::with_capacity(N);
        range_count.fill_zero().unwrap();
        Self {
            pow_count,
            range_count,
            cpu_checker,
        }
    }

    pub fn hybrid() -> Self {
        let cpu_checker = Some(Arc::new(PowerCheckerCpuTraceGenerator::default()));
        Self::new(cpu_checker)
    }

    pub fn pow_count_mut_ptr(&self) -> *mut u32 {
        self.pow_count.as_mut_ptr()
    }

    pub fn range_count_mut_ptr(&self) -> *mut u32 {
        self.range_count.as_mut_ptr()
    }

    pub fn cpu_checker(&self) -> Option<Arc<PowerCheckerCpuTraceGenerator<BASE, N>>> {
        self.cpu_checker.clone()
    }

    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        let mem = MemTracker::start("tracegen.pow_checker");
        let (cpu_pow_count, cpu_range_count) = if let Some(cpu_checker) = &self.cpu_checker {
            let (pow, range) = cpu_checker.take_counts();
            (
                Some(pow.as_slice().to_device().unwrap()),
                Some(range.as_slice().to_device().unwrap()),
            )
        } else {
            (None, None)
        };
        let trace = DeviceMatrix::with_capacity(N, PowerCheckerCols::<u8>::width());
        unsafe {
            pow_checker_tracegen(
                self.pow_count.as_ptr(),
                self.range_count.as_ptr(),
                cpu_pow_count.as_ref(),
                cpu_range_count.as_ref(),
                trace.buffer(),
                N,
            )
            .unwrap();
        }
        mem.emit_metrics();
        trace
    }
}
