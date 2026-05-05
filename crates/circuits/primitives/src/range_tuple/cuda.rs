use std::sync::{atomic::Ordering, Arc};

use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{cuda_abi::range_tuple::tracegen, range_tuple::RangeTupleCheckerChip, Chip};

pub struct RangeTupleCheckerChipGPU<const N: usize> {
    pub device_ctx: GpuDeviceCtx,
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<RangeTupleCheckerChip<N>>>,
    pub sizes: [u32; N],
}

impl<const N: usize> RangeTupleCheckerChipGPU<N> {
    pub fn new(sizes: [u32; N], device_ctx: GpuDeviceCtx) -> Self {
        assert!(N > 1, "RangeTupleChecker requires at least 2 dimensions");
        let range_max = sizes.iter().product::<u32>() as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity_on(range_max, &device_ctx));
        count.fill_zero_on(&device_ctx).unwrap();
        Self {
            device_ctx,
            count,
            cpu_chip: None,
            sizes,
        }
    }

    pub fn hybrid(cpu_chip: Arc<RangeTupleCheckerChip<N>>, device_ctx: GpuDeviceCtx) -> Self {
        let count = Arc::new(DeviceBuffer::<F>::with_capacity_on(
            cpu_chip.count.len(),
            &device_ctx,
        ));
        count.fill_zero_on(&device_ctx).unwrap();
        let sizes = *cpu_chip.sizes();
        Self {
            device_ctx,
            count,
            cpu_chip: Some(cpu_chip),
            sizes,
        }
    }
}

impl<RA, const N: usize> Chip<RA, GpuBackend> for RangeTupleCheckerChipGPU<N> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let cpu_count = self.cpu_chip.as_ref().map(|cpu_chip| {
            cpu_chip
                .count
                .iter()
                .map(|c| c.swap(0, Ordering::Relaxed))
                .collect::<Vec<_>>()
                .to_device_on(&self.device_ctx)
                .unwrap()
        });
        // ATTENTION: we create a new buffer to copy `count` into because this chip is stateful and
        // `count` will be reused.
        let trace = DeviceMatrix::<F>::with_capacity_on(self.count.len(), N + 1, &self.device_ctx);
        // Zero padding rows so stale pool data doesn't cause constraint violations.
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        let d_sizes = self.sizes.to_device_on(&self.device_ctx).unwrap();
        unsafe {
            tracegen(
                &self.count,
                &cpu_count,
                trace.buffer(),
                &d_sizes,
                self.device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        // Zero the internal count buffer because this chip is stateful and may be used again.
        self.count.fill_zero_on(&self.device_ctx).unwrap();
        AirProvingContext::simple_no_pis(trace)
    }

    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.count.len())
    }
}
