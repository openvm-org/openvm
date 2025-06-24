use std::sync::Arc;

use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, NUM_BITWISE_OP_LOOKUP_COLS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F, prover_backend::GpuBackend,
    types::SC,
};

use crate::{primitives::cuda::bitwise_op_lookup::tracegen, DeviceChip};

#[cfg(test)]
mod tests;

pub struct BitwiseOperationLookupChipGPU<const NUM_BITS: usize> {
    pub air: BitwiseOperationLookupAir<NUM_BITS>,
    pub count: Arc<DeviceBuffer<F>>,
}

impl<const NUM_BITS: usize> BitwiseOperationLookupChipGPU<NUM_BITS> {
    pub const fn num_rows() -> usize {
        1 << (2 * NUM_BITS)
    }

    pub fn new(bus: BitwiseOperationLookupBus) -> Self {
        // The first 2^(2 * NUM_BITS) indices are for range checking, the rest are for XOR
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(
            NUM_BITWISE_OP_LOOKUP_COLS * Self::num_rows(),
        ));
        count.fill_zero().unwrap();
        Self {
            air: BitwiseOperationLookupAir::new(bus),
            count,
        }
    }
}

impl<const NUM_BITS: usize> ChipUsageGetter for BitwiseOperationLookupChipGPU<NUM_BITS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        Self::num_rows()
    }

    fn trace_width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }
}

impl<const NUM_BITS: usize> DeviceChip<SC, GpuBackend> for BitwiseOperationLookupChipGPU<NUM_BITS> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        debug_assert_eq!(
            Self::num_rows() * NUM_BITWISE_OP_LOOKUP_COLS,
            self.count.len()
        );
        let trace = DeviceMatrix::<F>::new(
            self.count.clone(),
            Self::num_rows(),
            NUM_BITWISE_OP_LOOKUP_COLS,
        );
        unsafe {
            tracegen(&self.count, trace.buffer(), NUM_BITS as u32).unwrap();
        }
        trace
    }
}
