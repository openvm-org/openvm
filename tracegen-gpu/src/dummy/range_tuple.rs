use std::sync::Arc;

use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
};

use crate::{
    dummy::cuda::range_tuple_dummy::tracegen, primitives::range_tuple::RangeTupleCheckerChipGPU,
};

pub struct DummyInteractionChipGPU<const N: usize> {
    pub air: DummyInteractionAir,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<N>>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, tuple...]
impl<const N: usize> DummyInteractionChipGPU<N> {
    pub fn new(range_tuple_checker: Arc<RangeTupleCheckerChipGPU<N>>, data: Vec<u32>) -> Self {
        Self {
            air: DummyInteractionAir::new(N, true, range_tuple_checker.air.bus.inner.index),
            range_tuple_checker,
            data: data.to_device().unwrap(),
        }
    }

    pub fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace = DeviceMatrix::<F>::with_capacity(self.data.len() / N, N + 1);
        let sizes = self.range_tuple_checker.air.bus.sizes.to_device().unwrap();
        unsafe {
            tracegen(
                &self.data,
                trace.buffer(),
                &self.range_tuple_checker.count,
                &sizes,
            )
            .unwrap();
        }
        trace
    }
}
