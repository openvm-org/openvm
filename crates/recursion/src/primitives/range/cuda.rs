use openvm_cuda_backend::{base::DeviceMatrix, prelude::F};
use openvm_cuda_common::{copy::cuda_memcpy_on, memory_manager::MemTracker, stream::GpuDeviceCtx};

use crate::primitives::{cuda_abi::range_checker_tracegen, range::RangeCheckerCols};

pub struct RangeCheckerGpuTraceGenerator<const NUM_BITS: usize> {
    trace: DeviceMatrix<F>,
    device_ctx: GpuDeviceCtx,
}

impl<const NUM_BITS: usize> RangeCheckerGpuTraceGenerator<NUM_BITS> {
    pub fn new(device_ctx: GpuDeviceCtx) -> Self {
        let trace = DeviceMatrix::with_capacity_on(
            1 << NUM_BITS,
            RangeCheckerCols::<u8>::width(),
            &device_ctx,
        );
        trace.buffer().fill_zero_on(&device_ctx).unwrap();
        Self { trace, device_ctx }
    }

    pub fn from_vals(vals: &[usize], device_ctx: GpuDeviceCtx) -> Self {
        let res = Self::new(device_ctx);
        if vals.is_empty() {
            return res;
        }

        let mut count = vec![0u32; 1 << NUM_BITS];
        for &v in vals {
            count[v] += 1;
        }

        unsafe {
            cuda_memcpy_on::<false, true>(
                res.count_mut_ptr().cast(),
                count.as_ptr().cast(),
                std::mem::size_of_val(count.as_slice()),
                &res.device_ctx,
            )
            .unwrap();
        }
        res
    }

    pub fn count_ptr(&self) -> *const u32 {
        self.trace.buffer().as_ptr().wrapping_add(1 << NUM_BITS) as *const u32
    }

    pub fn count_mut_ptr(&self) -> *mut u32 {
        self.trace.buffer().as_mut_ptr().wrapping_add(1 << NUM_BITS) as *mut u32
    }

    pub fn generate_trace(self) -> DeviceMatrix<F> {
        let mem = MemTracker::start("tracegen.range_checker");
        unsafe {
            range_checker_tracegen(
                self.count_ptr(),
                self.trace.buffer(),
                NUM_BITS,
                self.device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        mem.emit_metrics();
        self.trace
    }
}
