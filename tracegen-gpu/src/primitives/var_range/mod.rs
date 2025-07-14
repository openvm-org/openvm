use std::sync::{atomic::Ordering, Arc};

use openvm_circuit_primitives::var_range::{
    VariableRangeCheckerAir, VariableRangeCheckerBus, VariableRangeCheckerChip,
    NUM_VARIABLE_RANGE_COLS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

use crate::{primitives::cuda::var_range::tracegen, DeviceChip};

#[cfg(test)]
mod tests;

pub struct VariableRangeCheckerChipGPU {
    pub air: VariableRangeCheckerAir,
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<VariableRangeCheckerChip>>,
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
            cpu_chip: None,
        }
    }

    pub fn hybrid(cpu_chip: Arc<VariableRangeCheckerChip>) -> Self {
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(cpu_chip.count.len()));
        count.fill_zero().unwrap();
        Self {
            air: cpu_chip.air,
            count,
            cpu_chip: Some(cpu_chip),
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
        let cpu_count = self.cpu_chip.as_ref().map(|cpu_chip| {
            cpu_chip
                .count
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
                .to_device()
                .unwrap()
        });
        let trace = DeviceMatrix::<F>::new(
            self.count.clone(),
            self.count.len(),
            NUM_VARIABLE_RANGE_COLS,
        );
        unsafe {
            tracegen(&self.count, &cpu_count, trace.buffer()).unwrap();
        }
        trace
    }
}
