use std::sync::{atomic::Ordering, Arc};

use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer, stream::DeviceContext};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    bitwise_op_lookup::{
        BitwiseOperationLookupChip, BitwiseOperationLookupCols, NUM_BITWISE_OP_LOOKUP_MULT_COLS,
    },
    cuda_abi::bitwise_op_lookup::tracegen,
    Chip,
};

pub struct BitwiseOperationLookupChipGPU<const NUM_BITS: usize> {
    pub device_ctx: DeviceContext,
    pub count: Arc<DeviceBuffer<F>>,
    pub cpu_chip: Option<Arc<BitwiseOperationLookupChip<NUM_BITS>>>,
}

impl<const NUM_BITS: usize> BitwiseOperationLookupChipGPU<NUM_BITS> {
    pub const fn num_rows() -> usize {
        1 << (2 * NUM_BITS)
    }

    pub fn new(device_ctx: DeviceContext) -> Self {
        // The first 2^(2 * NUM_BITS) indices are for range checking, the rest are for XOR
        let count = Arc::new(DeviceBuffer::<F>::with_capacity_on(
            NUM_BITWISE_OP_LOOKUP_MULT_COLS * Self::num_rows(),
            &device_ctx,
        ));
        count.fill_zero_on(&device_ctx).unwrap();
        Self {
            device_ctx,
            count,
            cpu_chip: None,
        }
    }

    pub fn hybrid(
        cpu_chip: Arc<BitwiseOperationLookupChip<NUM_BITS>>,
        device_ctx: DeviceContext,
    ) -> Self {
        assert_eq!(cpu_chip.count_range.len(), Self::num_rows());
        assert_eq!(cpu_chip.count_xor.len(), Self::num_rows());
        let count = Arc::new(DeviceBuffer::<F>::with_capacity_on(
            NUM_BITWISE_OP_LOOKUP_MULT_COLS * Self::num_rows(),
            &device_ctx,
        ));
        count.fill_zero_on(&device_ctx).unwrap();
        Self {
            device_ctx,
            count,
            cpu_chip: Some(cpu_chip),
        }
    }
}

impl<RA, const NUM_BITS: usize> Chip<RA, GpuBackend> for BitwiseOperationLookupChipGPU<NUM_BITS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let num_cols = BitwiseOperationLookupCols::<F, NUM_BITS>::width();
        debug_assert_eq!(
            NUM_BITWISE_OP_LOOKUP_MULT_COLS * Self::num_rows(),
            self.count.len()
        );
        let cpu_count = self.cpu_chip.as_ref().map(|cpu_chip| {
            cpu_chip
                .count_range
                .iter()
                .chain(cpu_chip.count_xor.iter())
                .map(|c| c.swap(0, Ordering::Relaxed))
                .collect::<Vec<_>>()
                .to_device_on(&self.device_ctx)
                .unwrap()
        });
        // ATTENTION: we create a new buffer to copy `count` into because this chip is stateful and
        // `count` will be reused.
        let trace =
            DeviceMatrix::<F>::with_capacity_on(Self::num_rows(), num_cols, &self.device_ctx);
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        unsafe {
            tracegen(
                &self.count,
                &cpu_count,
                trace.buffer(),
                NUM_BITS as u32,
                self.device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        // Zero the internal count buffer because this chip is stateful and may be used again.
        self.count.fill_zero_on(&self.device_ctx).unwrap();
        AirProvingContext::simple_no_pis(trace)
    }

    fn constant_trace_height(&self) -> Option<usize> {
        Some(Self::num_rows())
    }
}
