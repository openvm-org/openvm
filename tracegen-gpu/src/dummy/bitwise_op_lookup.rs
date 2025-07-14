use std::sync::Arc;

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
    dummy::cuda::bitwise_op_lookup::tracegen,
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU, DeviceChip,
};

pub struct DummyInteractionChipGPU<const NUM_BITS: usize> {
    pub air: DummyInteractionAir,
    pub bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, value, bits]
impl<const NUM_BITS: usize> DummyInteractionChipGPU<NUM_BITS> {
    pub fn new(bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>, data: Vec<u32>) -> Self {
        Self {
            air: DummyInteractionAir::new(4, true, bitwise.air.bus.inner.index),
            bitwise,
            data: data.to_device().unwrap(),
        }
    }
}

impl<const NUM_BITS: usize> ChipUsageGetter for DummyInteractionChipGPU<NUM_BITS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len() / 3
    }

    fn trace_width(&self) -> usize {
        self.air.field_width() + 1
    }
}

impl<const NUM_BITS: usize> DeviceChip<SC, GpuBackend> for DummyInteractionChipGPU<NUM_BITS> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace =
            DeviceMatrix::<F>::with_capacity(self.current_trace_height(), self.trace_width());
        unsafe {
            tracegen(
                trace.buffer(),
                self.current_trace_height(),
                &self.data,
                &self.bitwise.count,
                NUM_BITS as u32,
            )
            .unwrap();
        }
        trace
    }
}
