use std::sync::Arc;

use cuda_kernels::dummy::dummy_chip::tracegen;
use cuda_utils::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_circuit_primitives::var_range::NUM_VARIABLE_RANGE_PREPROCESSED_COLS;
use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use stark_backend_gpu::{base::DeviceMatrix, prelude::F};

use crate::primitives::{var_range::VariableRangeCheckerChipGPU,};

pub struct DummyInteractionChipGPU {
    pub air: DummyInteractionAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, value, bits]
impl DummyInteractionChipGPU {
    pub fn new(range_checker: Arc<VariableRangeCheckerChipGPU>, data: Vec<u32>) -> Self {
        Self {
            air: DummyInteractionAir::new(
                NUM_VARIABLE_RANGE_PREPROCESSED_COLS,
                true,
                range_checker.air.bus.index(),
            ),
            range_checker,
            data: data.to_device().unwrap(),
        }
    }

    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace = DeviceMatrix::<F>::with_capacity(self.data.len(), self.air.field_width() + 1);
        unsafe {
            tracegen(&self.data, trace.buffer(), &self.range_checker.count).unwrap();
        }
        trace
    }
}

