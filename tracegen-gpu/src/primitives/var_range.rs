use std::sync::Arc;

use cuda_kernels::primitives::var_range::tracegen;
use cuda_utils::d_buffer::DeviceBuffer;
use openvm_circuit_primitives::var_range::{
    VariableRangeCheckerAir, VariableRangeCheckerBus, NUM_VARIABLE_RANGE_COLS,
};
use stark_backend_gpu::{base::DeviceMatrix, prelude::F};

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
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    /// Inplace generates the trace.
    pub fn generate_trace(&self) -> DeviceMatrix<F> {
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
