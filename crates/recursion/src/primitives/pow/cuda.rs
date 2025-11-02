use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;

use crate::primitives::{cuda_abi::pow_checker_tracegen, pow::PowerCheckerCols};

#[derive(Debug)]
pub struct PowerCheckerGpuTraceGenerator<const BASE: usize, const N: usize> {
    trace: DeviceMatrix<F>,
}

impl<const BASE: usize, const N: usize> Default for PowerCheckerGpuTraceGenerator<BASE, N> {
    fn default() -> Self {
        let trace = DeviceMatrix::with_capacity(N, PowerCheckerCols::<u8>::width());
        trace.buffer().fill_zero().unwrap();
        Self { trace }
    }
}

impl<const BASE: usize, const N: usize> PowerCheckerGpuTraceGenerator<BASE, N> {
    pub fn pow_count_ptr(&self) -> *const u32 {
        self.trace.buffer().as_ptr().wrapping_add(2 * N) as *const u32
    }

    pub fn pow_count_mut_ptr(&self) -> *mut u32 {
        self.trace.buffer().as_mut_ptr().wrapping_add(2 * N) as *mut u32
    }

    pub fn range_count_ptr(&self) -> *const u32 {
        self.trace.buffer().as_ptr().wrapping_add(3 * N) as *const u32
    }

    pub fn range_count_mut_ptr(&self) -> *mut u32 {
        self.trace.buffer().as_mut_ptr().wrapping_add(3 * N) as *mut u32
    }

    pub fn generate_trace(self) -> DeviceMatrix<F> {
        unsafe {
            pow_checker_tracegen(
                self.pow_count_ptr(),
                self.range_count_ptr(),
                self.trace.buffer(),
                N,
            )
            .unwrap();
        }
        self.trace
    }
}
