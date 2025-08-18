use std::sync::{atomic::Ordering, Arc};

use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    cuda_abi::range_tuple::tracegen,
    range_tuple::{RangeTupleCheckerChip, NUM_RANGE_TUPLE_COLS},
};

pub struct RangeTupleCheckerChipGPU<const N: usize> {
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<RangeTupleCheckerChip<N>>>,
    pub sizes: [u32; N],
}

impl<const N: usize> RangeTupleCheckerChipGPU<N> {
    pub fn new(sizes: [u32; N]) -> Self {
        let range_max = sizes.iter().product::<u32>() as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(range_max));
        count.fill_zero().unwrap();
        Self {
            count,
            cpu_chip: None,
            sizes,
        }
    }

    pub fn hybrid(cpu_chip: Arc<RangeTupleCheckerChip<N>>) -> Self {
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(cpu_chip.count.len()));
        count.fill_zero().unwrap();
        let sizes = *cpu_chip.sizes();
        Self {
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
                .to_device()
                .unwrap()
        });
        // ATTENTION: we create a new buffer to copy `count` into because this chip is stateful and
        // `count` will be reused.
        let trace = DeviceMatrix::<F>::with_capacity(self.count.len(), NUM_RANGE_TUPLE_COLS);
        unsafe {
            tracegen(&self.count, &cpu_count, trace.buffer()).unwrap();
        }
        // Zero the internal count buffer because this chip is stateful and may be used again.
        self.count.fill_zero().unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}
