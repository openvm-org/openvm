use std::sync::Arc;

use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
};

use crate::{cuda_abi::range_tuple::dummy_tracegen, range_tuple::cuda::RangeTupleCheckerChipGPU};

pub struct DummyInteractionChipGPU<const N: usize> {
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<N>>,
    pub data: DeviceBuffer<u32>,
}

/// Expects trace to be: [1, tuple...]
impl<const N: usize> DummyInteractionChipGPU<N> {
    // TODO[stephenh]: Remove this allow once CUDA tests are uncommented
    #[allow(unused)]
    pub fn new(range_tuple_checker: Arc<RangeTupleCheckerChipGPU<N>>, data: Vec<u32>) -> Self {
        assert!(!data.is_empty());
        Self {
            range_tuple_checker,
            data: data.to_device().unwrap(),
        }
    }
}

impl<RA, const N: usize> Chip<RA, GpuBackend> for DummyInteractionChipGPU<N> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = self.data.len() / N;
        let width = N + 1;
        let trace = DeviceMatrix::<F>::with_capacity(height, width);
        let sizes = self.range_tuple_checker.sizes.to_device().unwrap();
        unsafe {
            dummy_tracegen(
                &self.data,
                trace.buffer(),
                &self.range_tuple_checker.count,
                &sizes,
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
