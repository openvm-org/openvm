use std::sync::Arc;

use openvm_circuit_primitives::var_range::NUM_VARIABLE_RANGE_PREPROCESSED_COLS;
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

use crate::{
    dummy::cuda::dummy_chip::tracegen, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

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
}

impl ChipUsageGetter for DummyInteractionChipGPU {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        self.air.field_width() + 1
    }
}

impl DeviceChip<SC, GpuBackend> for DummyInteractionChipGPU {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace =
            DeviceMatrix::<F>::with_capacity(self.current_trace_height(), self.trace_width());
        unsafe {
            tracegen(&self.data, trace.buffer(), &self.range_checker.count).unwrap();
        }
        trace
    }
}
