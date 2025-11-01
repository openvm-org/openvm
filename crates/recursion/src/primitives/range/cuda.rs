use cuda_backend_v2::GpuBackendV2;
use openvm_cuda_backend::base::DeviceMatrix;
use stark_backend_v2::{F, prover::AirProvingContextV2};

use crate::primitives::{cuda_abi::range_checker_tracegen, range::RangeCheckerCols};

#[derive(Debug)]
pub struct RangeCheckerGpuTraceGenerator<const NUM_BITS: usize> {
    trace: DeviceMatrix<F>,
}

impl<const NUM_BITS: usize> Default for RangeCheckerGpuTraceGenerator<NUM_BITS> {
    fn default() -> Self {
        let trace = DeviceMatrix::with_capacity(1 << NUM_BITS, RangeCheckerCols::<u8>::width());
        trace.buffer().fill_zero().unwrap();
        Self { trace }
    }
}

impl<const NUM_BITS: usize> RangeCheckerGpuTraceGenerator<NUM_BITS> {
    pub fn count_ptr(&self) -> *const u32 {
        self.trace.buffer().as_ptr().wrapping_add(1 << NUM_BITS) as *const u32
    }

    pub fn count_mut_ptr(&self) -> *mut u32 {
        self.trace.buffer().as_mut_ptr().wrapping_add(1 << NUM_BITS) as *mut u32
    }

    pub fn generate_proving_ctx(&self) -> AirProvingContextV2<GpuBackendV2> {
        unsafe {
            range_checker_tracegen(self.count_ptr(), self.trace.buffer(), NUM_BITS).unwrap();
        }
        AirProvingContextV2::simple_no_pis(self.trace.clone())
    }
}
