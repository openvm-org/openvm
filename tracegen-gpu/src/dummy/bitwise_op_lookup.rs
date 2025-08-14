use std::sync::Arc;

use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
};

use crate::{
    dummy::cuda::bitwise_op_lookup::tracegen,
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
};

const RECORD_WIDTH: usize = 3;
const NUM_COLS: usize = 5;

pub struct DummyInteractionChipGPU<const NUM_BITS: usize> {
    pub bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, x, y, z, op]
impl<const NUM_BITS: usize> DummyInteractionChipGPU<NUM_BITS> {
    pub fn new(bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>, data: Vec<u32>) -> Self {
        assert!(!data.is_empty());
        Self {
            bitwise,
            data: data.to_device().unwrap(),
        }
    }
}

impl<RA, const NUM_BITS: usize> Chip<RA, GpuBackend> for DummyInteractionChipGPU<NUM_BITS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = self.data.len() / RECORD_WIDTH;
        let trace = DeviceMatrix::<F>::with_capacity(height, NUM_COLS);
        unsafe {
            tracegen(
                trace.buffer(),
                &self.data,
                &self.bitwise.count,
                NUM_BITS as u32,
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
