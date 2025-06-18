use std::sync::Arc;

use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, NUM_BITWISE_OP_LOOKUP_COLS,
};
use stark_backend_gpu::{base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F};

use crate::primitives::cuda::bitwise_op_lookup::tracegen;

#[cfg(test)]
mod tests;

pub struct BitwiseOperationLookupChipGPU<const NUM_BITS: usize> {
    pub air: BitwiseOperationLookupAir<NUM_BITS>,
    pub count: Arc<DeviceBuffer<F>>,
}

impl<const NUM_BITS: usize> BitwiseOperationLookupChipGPU<NUM_BITS> {
    pub const fn num_rows() -> usize {
        1 << (2 * NUM_BITS)
    }

    pub fn new(bus: BitwiseOperationLookupBus) -> Self {
        // The first 2^(2 * NUM_BITS) indices are for range checking, the rest are for XOR
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(
            NUM_BITWISE_OP_LOOKUP_COLS * Self::num_rows(),
        ));
        count.fill_zero().unwrap();
        Self {
            air: BitwiseOperationLookupAir::new(bus),
            count,
        }
    }

    /// Inplace generates the trace.
    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        debug_assert_eq!(
            Self::num_rows() * NUM_BITWISE_OP_LOOKUP_COLS,
            self.count.len()
        );
        let trace = DeviceMatrix::<F>::new(
            self.count.clone(),
            Self::num_rows(),
            NUM_BITWISE_OP_LOOKUP_COLS,
        );
        unsafe {
            tracegen(&self.count, trace.buffer(), NUM_BITS as u32).unwrap();
        }
        trace
    }
}
