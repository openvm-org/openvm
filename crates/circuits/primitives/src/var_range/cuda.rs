use std::sync::{atomic::Ordering, Arc};

use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    cuda_abi::var_range::tracegen,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip, NUM_VARIABLE_RANGE_COLS},
};

pub struct VariableRangeCheckerChipGPU {
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
            count,
            cpu_chip: None,
        }
    }

    pub fn hybrid(cpu_chip: Arc<VariableRangeCheckerChip>) -> Self {
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(cpu_chip.count.len()));
        count.fill_zero().unwrap();
        Self {
            count,
            cpu_chip: Some(cpu_chip),
        }
    }
}

impl<RA> Chip<RA, GpuBackend> for VariableRangeCheckerChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        assert_eq!(size_of::<F>(), size_of::<u32>());
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
        let trace = DeviceMatrix::<F>::with_capacity(self.count.len(), NUM_VARIABLE_RANGE_COLS);
        unsafe {
            tracegen(&self.count, &cpu_count, trace.buffer()).unwrap();
        }
        // Zero the internal count buffer because this chip is stateful and may be used again.
        self.count.fill_zero().unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}
