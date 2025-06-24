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
    dummy::cuda::range_tuple_dummy::tracegen, primitives::range_tuple::RangeTupleCheckerChipGPU,
    DeviceChip,
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
}

impl<const N: usize> ChipUsageGetter for DummyInteractionChipGPU<N> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len() / N
    }

    fn trace_width(&self) -> usize {
        N + 1
    }
}

impl<const N: usize> DeviceChip<SC, GpuBackend> for DummyInteractionChipGPU<N> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let trace =
            DeviceMatrix::<F>::with_capacity(self.current_trace_height(), self.trace_width());
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
