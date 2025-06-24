use std::sync::Arc;

use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, NUM_RANGE_TUPLE_COLS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F, prover_backend::GpuBackend,
    types::SC,
};

use crate::{primitives::cuda::range_tuple::tracegen, DeviceChip};

#[cfg(test)]
mod tests;

pub struct RangeTupleCheckerChipGPU<const N: usize> {
    pub air: RangeTupleCheckerAir<N>,
    pub count: Arc<DeviceBuffer<F>>,
}

impl<const N: usize> RangeTupleCheckerChipGPU<N> {
    pub fn new(bus: RangeTupleCheckerBus<N>) -> Self {
        let range_max = bus.sizes.iter().product::<u32>() as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(range_max));
        count.fill_zero().unwrap();
        Self {
            air: RangeTupleCheckerAir { bus },
            count,
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
        let trace =
            DeviceMatrix::<F>::new(self.count.clone(), self.count.len(), NUM_RANGE_TUPLE_COLS);
        unsafe {
            tracegen(&self.count, trace.buffer()).unwrap();
        }
        trace
    }
}
