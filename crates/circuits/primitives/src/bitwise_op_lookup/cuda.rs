use std::sync::{atomic::Ordering, Arc};

use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    bitwise_op_lookup::{BitwiseOperationLookupChip, NUM_BITWISE_OP_LOOKUP_COLS},
    cuda_abi::bitwise_op_lookup::tracegen,
};

pub struct BitwiseOperationLookupChipGPU<const NUM_BITS: usize> {
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<BitwiseOperationLookupChip<NUM_BITS>>>,
}

impl<const NUM_BITS: usize> BitwiseOperationLookupChipGPU<NUM_BITS> {
    pub const fn num_rows() -> usize {
        1 << (2 * NUM_BITS)
    }

    pub fn new() -> Self {
        // The first 2^(2 * NUM_BITS) indices are for range checking, the rest are for XOR
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(
            NUM_BITWISE_OP_LOOKUP_COLS * Self::num_rows(),
        ));
        count.fill_zero().unwrap();
        Self {
            count,
            cpu_chip: None,
        }
    }

    pub fn hybrid(cpu_chip: Arc<BitwiseOperationLookupChip<NUM_BITS>>) -> Self {
        assert_eq!(cpu_chip.count_range.len(), Self::num_rows());
        assert_eq!(cpu_chip.count_xor.len(), Self::num_rows());
        let count = Arc::new(DeviceBuffer::<F>::with_capacity(
            NUM_BITWISE_OP_LOOKUP_COLS * Self::num_rows(),
        ));
        count.fill_zero().unwrap();
        Self {
            count,
            cpu_chip: Some(cpu_chip),
        }
    }
}

impl<const NUM_BITS: usize> Default for BitwiseOperationLookupChipGPU<NUM_BITS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<RA, const NUM_BITS: usize> Chip<RA, GpuBackend> for BitwiseOperationLookupChipGPU<NUM_BITS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        debug_assert_eq!(
            Self::num_rows() * NUM_BITWISE_OP_LOOKUP_COLS,
            self.count.len()
        );
        let cpu_count = self.cpu_chip.as_ref().map(|cpu_chip| {
            cpu_chip
                .count_range
                .iter()
                .chain(cpu_chip.count_xor.iter())
                .map(|c| c.swap(0, Ordering::Relaxed))
                .collect::<Vec<_>>()
                .to_device()
                .unwrap()
        });
        // ATTENTION: we create a new buffer to copy `count` into because this chip is stateful and
        // `count` will be reused.
        let trace = DeviceMatrix::<F>::with_capacity(Self::num_rows(), NUM_BITWISE_OP_LOOKUP_COLS);
        unsafe {
            tracegen(&self.count, &cpu_count, trace.buffer(), NUM_BITS as u32).unwrap();
        }
        // Zero the internal count buffer because this chip is stateful and may be used again.
        self.count.fill_zero().unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}
