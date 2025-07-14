use std::sync::{atomic::Ordering, Arc};

use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip, NUM_RANGE_TUPLE_COLS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

use crate::{primitives::cuda::range_tuple::tracegen, DeviceChip};

#[cfg(test)]
mod tests;

pub struct RangeTupleCheckerChipGPU<const N: usize> {
    pub air: RangeTupleCheckerAir<N>,
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<RangeTupleCheckerChip<N>>>,
}

impl<const N: usize> RangeTupleCheckerChipGPU<N> {
    pub fn new(bus: RangeTupleCheckerBus<N>) -> Self {
        let range_max = bus.sizes.iter().product::<u32>() as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(range_max));
        count.fill_zero().unwrap();
        Self {
            air: RangeTupleCheckerAir { bus },
            count,
            cpu_chip: None,
        }
    }

    pub fn hybrid(cpu_chip: Arc<RangeTupleCheckerChip<N>>) -> Self {
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(cpu_chip.count.len()));
        count.fill_zero().unwrap();
        Self {
            air: cpu_chip.air,
            count,
            cpu_chip: Some(cpu_chip),
        }
    }
}

impl<const N: usize> ChipUsageGetter for RangeTupleCheckerChipGPU<N> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.count.len()
    }

    fn trace_width(&self) -> usize {
        NUM_RANGE_TUPLE_COLS
    }
}

impl<const N: usize> DeviceChip<SC, GpuBackend> for RangeTupleCheckerChipGPU<N> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
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
        trace
    }
}
