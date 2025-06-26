use std::sync::Arc;

use openvm_circuit::{
    system::poseidon2::air::Poseidon2PeripheryAir, utils::next_power_of_two_or_zero,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::{
    interaction::LookupBus, p3_air::BaseAir, prover::hal::MatrixDimensions, rap::get_air_name,
    AirRef, ChipUsageGetter,
};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::d_buffer::DeviceBuffer, prelude::F, prover_backend::GpuBackend,
    types::SC,
};

use crate::{system::cuda::poseidon2, DeviceChip};

pub struct SharedBuffer<T> {
    pub buffer: Arc<DeviceBuffer<T>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

pub struct Poseidon2ChipGPU<const SBOX_REGISTERS: usize> {
    pub air: Arc<Poseidon2PeripheryAir<F, SBOX_REGISTERS>>,
    pub records: Arc<DeviceBuffer<F>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

impl<const SBOX_REGISTERS: usize> Poseidon2ChipGPU<SBOX_REGISTERS> {
    pub fn new(config: Poseidon2Config<F>, bus: LookupBus, max_buffer_size: usize) -> Self {
        let subair = Arc::new(Poseidon2SubAir::new(config.constants.into()));
        let idx = DeviceBuffer::<u32>::with_capacity(1);
        idx.fill_zero().unwrap();
        Self {
            air: Arc::new(Poseidon2PeripheryAir::new(subair, bus)),
            records: Arc::new(DeviceBuffer::<F>::with_capacity(max_buffer_size)),
            idx: Arc::new(idx),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        SharedBuffer {
            buffer: self.records.clone(),
            idx: self.idx.clone(),
        }
    }
}

impl<const SBOX_REGISTERS: usize> ChipUsageGetter for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        0
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}

impl<const SBOX_REGISTERS: usize> DeviceChip<SC, GpuBackend> for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn air(&self) -> AirRef<SC> {
        self.air.clone()
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let mut num_records = self.records.len();
        let counts = DeviceBuffer::<u32>::with_capacity(num_records);
        unsafe {
            poseidon2::deduplicate_records(&self.records, &counts, &mut num_records)
                .expect("Failed to deduplicate records");
        }
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &self.records,
                &counts,
                num_records,
                SBOX_REGISTERS,
            )
            .expect("Failed to generate trace");
        }
        trace
    }
}
