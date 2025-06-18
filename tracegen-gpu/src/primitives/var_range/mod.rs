use std::sync::Arc;

use openvm_circuit_primitives::var_range::{
    VariableRangeCheckerAir, VariableRangeCheckerBus, NUM_VARIABLE_RANGE_COLS,
};
use stark_backend_gpu::{base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F};

use crate::primitives::cuda::var_range::tracegen;

#[cfg(test)]
mod tests;

pub struct VariableRangeCheckerChipGPU {
    pub air: VariableRangeCheckerAir,
    pub count: Arc<DeviceBuffer<F>>,
}

/// [value, bits] are in preprocessed trace
/// generate_trace returns [count]
impl VariableRangeCheckerChipGPU {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        let num_rows = (1 << (bus.range_max_bits + 1)) as usize;
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(num_rows));
        count.fill_zero().unwrap();
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    /// Inplace generates the trace.
    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        assert_eq!(size_of::<F>(), size_of::<u32>());
        let trace = DeviceMatrix::<F>::new(
            self.count.clone(),
            self.count.len(),
            NUM_VARIABLE_RANGE_COLS,
        );
        unsafe {
            tracegen(&self.count, trace.buffer()).unwrap();
        }
        trace
    }
}
