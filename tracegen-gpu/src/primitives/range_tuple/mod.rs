use std::sync::{atomic::Ordering, Arc};

use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerBus, RangeTupleCheckerChip, NUM_RANGE_TUPLE_COLS,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prover_backend::GpuBackend,
    types::F,
};

use crate::primitives::cuda::range_tuple::tracegen;

#[cfg(test)]
mod tests;

pub struct RangeTupleCheckerChipGPU<const N: usize> {
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<RangeTupleCheckerChip<N>>>,
    pub sizes: [u32; N],
}

impl<const N: usize> RangeTupleCheckerChipGPU<N> {
    pub fn new(bus: RangeTupleCheckerBus<N>) -> Self {
        let range_max = bus.sizes.iter().product::<u32>() as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(range_max));
        count.fill_zero().unwrap();
        Self {
            count,
            cpu_chip: None,
            sizes: bus.sizes,
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
                .map(|c| c.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
                .to_device()
                .unwrap()
        });
        let trace =
            DeviceMatrix::<F>::new(self.count.clone(), self.count.len(), NUM_RANGE_TUPLE_COLS);
        unsafe {
            tracegen(&self.count, &cpu_count, trace.buffer()).unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
