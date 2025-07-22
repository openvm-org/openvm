use std::sync::Arc;

use openvm_circuit_primitives::var_range::NUM_VARIABLE_RANGE_PREPROCESSED_COLS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
};

use crate::{dummy::cuda::var_range::tracegen, primitives::var_range::VariableRangeCheckerChipGPU};

pub struct DummyInteractionChipGPU {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, value, bits]
impl DummyInteractionChipGPU {
    pub fn new(range_checker: Arc<VariableRangeCheckerChipGPU>, data: Vec<u32>) -> Self {
        assert!(!data.is_empty());
        Self {
            range_checker,
            data: data.to_device().unwrap(),
        }
    }
}

impl<RA> Chip<RA, GpuBackend> for DummyInteractionChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = self.data.len();
        let width = NUM_VARIABLE_RANGE_PREPROCESSED_COLS + 1;
        let trace = DeviceMatrix::<F>::with_capacity(height, width);
        unsafe {
            tracegen(&self.data, trace.buffer(), &self.range_checker.count).unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
