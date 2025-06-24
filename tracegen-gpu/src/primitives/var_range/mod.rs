use std::sync::Arc;

use openvm_circuit_primitives::var_range::{
    VariableRangeCheckerAir, VariableRangeCheckerBus, NUM_VARIABLE_RANGE_COLS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F, prover_backend::GpuBackend,
    types::SC,
};

use crate::{primitives::cuda::var_range::tracegen, DeviceChip};

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
}

impl ChipUsageGetter for VariableRangeCheckerChipGPU {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.count.len()
    }

    fn trace_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }
}

impl DeviceChip<SC, GpuBackend> for VariableRangeCheckerChipGPU {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
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
