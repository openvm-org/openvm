use std::sync::Arc;

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    cuda_abi::poseidon2::{self, DeferralPoseidon2Count},
    poseidon2::DeferralPoseidon2Cols,
};

#[derive(Clone)]
pub struct DeferralPoseidon2SharedBuffer {
    pub records: Arc<DeviceBuffer<F>>,
    pub counts: Arc<DeviceBuffer<DeferralPoseidon2Count>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

pub struct DeferralPoseidon2ChipGpu {
    pub records: Arc<DeviceBuffer<F>>,
    pub counts: Arc<DeviceBuffer<DeferralPoseidon2Count>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    pub sbox_registers: usize,
}

impl DeferralPoseidon2ChipGpu {
    /// Creates a new deferral Poseidon2 chip with a device buffer of `max_buffer_size` field
    /// elements. Each Poseidon2 record occupies `POSEIDON2_WIDTH` (16) field elements, so the
    /// buffer can hold `max_buffer_size / POSEIDON2_WIDTH` records.
    pub fn new(max_buffer_size: usize, sbox_registers: usize) -> Self {
        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity(1));
        idx.fill_zero().unwrap();

        Self {
            records: Arc::new(DeviceBuffer::<F>::with_capacity(max_buffer_size)),
            counts: Arc::new(DeviceBuffer::<DeferralPoseidon2Count>::with_capacity(
                max_buffer_size,
            )),
            idx,
            sbox_registers,
        }
    }

    pub fn shared_buffer(&self) -> DeferralPoseidon2SharedBuffer {
        DeferralPoseidon2SharedBuffer {
            records: self.records.clone(),
            counts: self.counts.clone(),
            idx: self.idx.clone(),
        }
    }

    pub fn trace_width() -> usize {
        DeferralPoseidon2Cols::<F>::width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralPoseidon2ChipGpu {
    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let mut num_records = self.idx.to_host().unwrap()[0] as usize;
        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        unsafe {
            let d_num_records = [num_records].to_device().unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &self.records,
                &self.counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
            )
            .expect("Failed to get deferral poseidon2 temp bytes");

            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity(temp_bytes)
            };

            poseidon2::deduplicate_records(
                &self.records,
                &self.counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
            )
            .expect("Failed to deduplicate deferral poseidon2 records");

            num_records = *d_num_records.to_host().unwrap().first().unwrap();
        }

        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, Self::trace_width());

        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &self.records,
                &self.counts,
                num_records,
                self.sbox_registers,
            )
            .expect("Failed to generate deferral poseidon2 trace");
        }

        self.idx
            .fill_zero()
            .expect("Failed to reset deferral poseidon2 record index");

        AirProvingContext::simple_no_pis(trace)
    }
}

pub fn poseidon2_buffer_capacity(max_trace_height: usize) -> usize {
    max_trace_height.next_power_of_two() * 2 * (DIGEST_SIZE * 2)
}
