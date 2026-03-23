use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{count::DeferralCircuitCountCols, cuda_abi::count};

#[derive(new)]
pub struct DeferralCircuitCountChipGpu {
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralCircuitCountChipGpu {
    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        if self.num_deferral_circuits == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = DeferralCircuitCountCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(self.num_deferral_circuits);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            count::tracegen(
                trace.buffer(),
                trace_height,
                &self.count,
                self.num_deferral_circuits,
            )
            .expect("Failed to generate deferral count trace");
        }

        self.count
            .fill_zero()
            .expect("Failed to reset deferral count");
        AirProvingContext::simple_no_pis(trace)
    }
}
