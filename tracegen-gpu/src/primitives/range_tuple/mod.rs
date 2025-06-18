use std::sync::Arc;

use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, NUM_RANGE_TUPLE_COLS,
};
use stark_backend_gpu::{base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F};

use crate::primitives::cuda::range_tuple::tracegen;

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

    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace =
            DeviceMatrix::<F>::new(self.count.clone(), self.count.len(), NUM_RANGE_TUPLE_COLS);
        unsafe {
            tracegen(&self.count, trace.buffer()).unwrap();
        }
        trace
    }
}
