#[cfg(feature = "rvr")]
use std::mem::size_of;
use std::sync::Arc;

use derive_new::new;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::cuda::GpuPostflightError;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{count::DeferralCircuitCountCols, cuda_abi::count};

#[derive(new)]
pub struct DeferralCircuitCountChipGpu {
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub device_ctx: GpuDeviceCtx,
}

#[cfg(feature = "rvr")]
impl DeferralCircuitCountChipGpu {
    /// Generates the periphery trace directly from the shared histogram. No
    /// execution arena is consumed; the caller supplies the existing padded
    /// segment trace-height limit.
    pub fn generate_proving_ctx_direct(
        &self,
        max_trace_height: usize,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        if self.num_deferral_circuits == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_width = DeferralCircuitCountCols::<F>::width();
        let trace_height = self
            .num_deferral_circuits
            .checked_next_power_of_two()
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "Deferral Count trace height overflow".to_string(),
                )
            })?;
        if trace_height > max_trace_height {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "Deferral Count padded trace height {trace_height} exceeds segment limit {max_trace_height}"
            )));
        }
        trace_height
            .checked_mul(trace_width)
            .and_then(|elements| elements.checked_mul(size_of::<F>()))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "Deferral Count trace allocation overflow".to_string(),
                )
            })?;
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, &self.device_ctx);
        unsafe {
            count::tracegen(
                trace.buffer(),
                trace_height,
                &self.count,
                self.num_deferral_circuits,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        self.count.fill_zero_on(&self.device_ctx)?;
        Ok(AirProvingContext::simple_no_pis(trace))
    }
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
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, &self.device_ctx);

        unsafe {
            count::tracegen(
                trace.buffer(),
                trace_height,
                &self.count,
                self.num_deferral_circuits,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral count trace");
        }

        self.count
            .fill_zero_on(&self.device_ctx)
            .expect("Failed to reset deferral count");
        AirProvingContext::simple_no_pis(trace)
    }
}
