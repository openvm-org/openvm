use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::DeviceContext};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{count::DeferralCircuitCountCols, cuda_abi::count};

#[derive(new)]
pub struct DeferralCircuitCountChipGpu {
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub ctx: DeviceContext,
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralCircuitCountChipGpu {
    fn constant_trace_height(&self) -> Option<usize> {
        Some(next_power_of_two_or_zero(self.num_deferral_circuits))
    }

    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        if self.num_deferral_circuits == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = DeferralCircuitCountCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(self.num_deferral_circuits);
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, &self.ctx);

        unsafe {
            count::tracegen(
                trace.buffer(),
                trace_height,
                &self.count,
                self.num_deferral_circuits,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral count trace");
        }

        self.count
            .fill_zero_on(&self.ctx)
            .expect("Failed to reset deferral count");
        AirProvingContext::simple_no_pis(trace)
    }
}
