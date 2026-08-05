use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::prover::AirProvingContext;

use super::poseidon2::SharedBuffer;
use crate::{
    arch::{cuda::postflight::GpuPostflightError, PublicValuesState},
    cuda_abi::public_values,
    system::public_values::{
        public_values_cells, public_values_poseidon2_record_count, PublicValuesAir,
        PublicValuesCols, PublicValuesPvs,
    },
};

/// GPU trace generator for the append-only public-output accumulator.
pub struct PublicValuesChipGPU {
    pub air: PublicValuesAir,
    poseidon2_buffer: SharedBuffer<F>,
    device_ctx: GpuDeviceCtx,
}

impl PublicValuesChipGPU {
    pub fn new(
        air: PublicValuesAir,
        poseidon2_buffer: SharedBuffer<F>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        Self {
            air,
            poseidon2_buffer,
            device_ctx,
        }
    }

    /// Maximum records to reserve; only valid reveal rows emit records.
    pub fn poseidon2_record_count(&self) -> usize {
        public_values_poseidon2_record_count(self.air.num_public_value_cells)
    }

    pub fn generate_proving_ctx(
        &self,
        initial_len: usize,
        state: &PublicValuesState,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        self.validate_boundary(initial_len, state)?;

        let values = public_values_cells::<F>(state);
        let d_values = values.to_device_on(&self.device_ctx)?;
        let height = self.air.trace_height();
        let width = PublicValuesCols::<F>::width();
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, &self.device_ctx);
        let pvs =
            DeviceBuffer::<F>::with_capacity_on(PublicValuesPvs::<F>::width(), &self.device_ctx);
        let poseidon2_records = self.poseidon2_buffer.records();
        unsafe {
            public_values::tracegen(
                trace.buffer(),
                height,
                width,
                &d_values,
                initial_len,
                state.len(),
                &pvs,
                &poseidon2_records,
                &self.poseidon2_buffer.idx,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        let public_values = pvs.to_host_on(&self.device_ctx)?;
        Ok(AirProvingContext::simple(trace, public_values))
    }

    pub(crate) fn validate_boundary(
        &self,
        initial_len: usize,
        state: &PublicValuesState,
    ) -> Result<(), GpuPostflightError> {
        let max_values = self.air.trace_height();
        if state.max_public_values() != max_values {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "public-values capacity is {}, expected {max_values}",
                state.max_public_values()
            )));
        }
        if initial_len > state.len() {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "public-values length decreased from {initial_len} to {} within one segment",
                state.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit_primitives::Chip;
    use openvm_cuda_backend::data_transporter::assert_eq_host_and_device_matrix_col_maj;
    use openvm_cuda_common::{
        common::get_device,
        copy::MemCopyD2H,
        stream::{CudaStream, StreamGuard},
    };
    use openvm_stark_backend::{interaction::PermutationCheckBus, prover::ColMajorMatrix};
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;
    use crate::{
        arch::{vm_poseidon2_config, U16_CELLS_PER_PUBLIC_VALUE},
        system::{
            cuda::Poseidon2PeripheryChipGPU,
            poseidon2::Poseidon2PeripheryChip,
            public_values::{PublicValuesBus, PublicValuesChip},
        },
    };

    fn assert_gpu_matches_cpu(state: &PublicValuesState, initial_len: usize) {
        let air = PublicValuesAir::new(
            state.max_public_values() * U16_CELLS_PER_PUBLIC_VALUE,
            PublicValuesBus::new(0),
            PermutationCheckBus::new(1),
        );
        let cpu_hasher = Arc::new(Poseidon2PeripheryChip::new(vm_poseidon2_config(), 3));
        let cpu_ctx = PublicValuesChip::new(air.clone(), cpu_hasher)
            .generate_proving_ctx::<BabyBearPoseidon2Config>(state, initial_len);

        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let gpu_hasher = Arc::new(Poseidon2PeripheryChipGPU::new(1, device_ctx.clone()));
        let shared_buffer = gpu_hasher.shared_buffer();
        let gpu_chip = PublicValuesChipGPU::new(air, shared_buffer.clone(), device_ctx.clone());
        gpu_hasher.prepare_records(gpu_chip.poseidon2_record_count());
        let gpu_ctx = gpu_chip
            .generate_proving_ctx(initial_len, state)
            .expect("GPU public-values trace generation must succeed");

        assert_eq!(gpu_ctx.public_values, cpu_ctx.public_values);
        let cpu_trace = ColMajorMatrix::from_row_major(&cpu_ctx.common_main);
        assert_eq_host_and_device_matrix_col_maj(&cpu_trace, &gpu_ctx.common_main, &device_ctx);
        assert_eq!(
            shared_buffer.idx.to_host_on(&device_ctx).unwrap()[0] as usize,
            state.len() - initial_len,
            "public-values accumulator must append exactly one record per reveal"
        );
        let _ = gpu_hasher.generate_proving_ctx();
    }

    #[test]
    fn empty_stream_matches_cpu() {
        assert_gpu_matches_cpu(&PublicValuesState::new(4), 0);
    }

    #[test]
    fn zero_reveals_in_segment_matches_cpu() {
        let mut state = PublicValuesState::new(4);
        state.try_push(7).unwrap();
        assert_gpu_matches_cpu(&state, state.len());
    }

    #[test]
    fn continued_stream_matches_cpu() {
        let mut state = PublicValuesState::new(4);
        state.try_push(0x1122_3344_5566_7788).unwrap();
        state.try_push(0).unwrap();
        let initial_len = state.len();
        state.try_push(u64::MAX).unwrap();
        assert_gpu_matches_cpu(&state, initial_len);
    }
}
