use std::mem::size_of;

use derive_new::new;
use openvm_circuit::{
    arch::DenseRecordArena,
    primitives::Chip,
    system::phantom::{PhantomCols, PhantomRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};

use crate::cuda_abi::phantom;

#[derive(new)]
pub struct PhantomChipGPU {
    device_ctx: GpuDeviceCtx,
}

impl PhantomChipGPU {
    pub fn trace_height(arena: &DenseRecordArena) -> usize {
        let record_size = size_of::<PhantomRecord>();
        let records_len = arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    pub fn trace_width() -> usize {
        PhantomCols::<F>::width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for PhantomChipGPU {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let num_records = Self::trace_height(&arena);
        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            Self::trace_width(),
            &self.device_ctx,
        );
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        unsafe {
            phantom::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &arena.allocated().to_device_on(&self.device_ctx).unwrap(),
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate trace");
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
