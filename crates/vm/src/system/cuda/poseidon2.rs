#[cfg(feature = "metrics")]
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use openvm_circuit::{
    system::poseidon2::columns::Poseidon2PeripheryCols, utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, prover_backend::GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};

use crate::cuda_abi::poseidon2;

#[derive(Clone)]
pub struct SharedBuffer<T> {
    pub buffer: Arc<DeviceBuffer<T>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    #[cfg(feature = "metrics")]
    pub(crate) current_trace_height: Arc<AtomicUsize>,
}

pub struct Poseidon2ChipGPU<const SBOX_REGISTERS: usize> {
    pub records: Arc<DeviceBuffer<F>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    #[cfg(feature = "metrics")]
    pub(crate) current_trace_height: Arc<AtomicUsize>,
}

impl<const SBOX_REGISTERS: usize> Poseidon2ChipGPU<SBOX_REGISTERS> {
    pub fn new(max_buffer_size: usize) -> Self {
        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity(1));
        idx.fill_zero().unwrap();
        Self {
            records: Arc::new(DeviceBuffer::<F>::with_capacity(max_buffer_size)),
            idx,
            #[cfg(feature = "metrics")]
            current_trace_height: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        SharedBuffer {
            buffer: self.records.clone(),
            idx: self.idx.clone(),
            #[cfg(feature = "metrics")]
            current_trace_height: self.current_trace_height.clone(),
        }
    }

    pub fn trace_width() -> usize {
        Poseidon2PeripheryCols::<F, SBOX_REGISTERS>::width()
    }
}

impl<RA, const SBOX_REGISTERS: usize> Chip<RA, GpuBackend> for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let mut num_records = self.idx.to_host().unwrap()[0] as usize;
        let counts = DeviceBuffer::<u32>::with_capacity(num_records);
        unsafe {
            let d_num_records = [num_records].to_device().unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &self.records,
                &counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
            )
            .expect("Failed to get temp bytes");
            let d_temp_storage = DeviceBuffer::<u8>::with_capacity(temp_bytes);
            poseidon2::deduplicate_records(
                &self.records,
                &counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
            )
            .expect("Failed to deduplicate records");
            num_records = *d_num_records.to_host().unwrap().first().unwrap();
        }
        #[cfg(feature = "metrics")]
        self.current_trace_height
            .store(num_records, std::sync::atomic::Ordering::Relaxed);
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, Self::trace_width());
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
        // Reset state of this chip.
        self.idx.fill_zero().unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}

pub enum Poseidon2PeripheryChipGPU {
    Register0(Poseidon2ChipGPU<0>),
    Register1(Poseidon2ChipGPU<1>),
}

impl Poseidon2PeripheryChipGPU {
    pub fn new(max_buffer_size: usize, sbox_registers: usize) -> Self {
        match sbox_registers {
            0 => Self::Register0(Poseidon2ChipGPU::new(max_buffer_size)),
            1 => Self::Register1(Poseidon2ChipGPU::new(max_buffer_size)),
            _ => panic!("Invalid number of sbox registers: {}", sbox_registers),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        match self {
            Self::Register0(chip) => chip.shared_buffer(),
            Self::Register1(chip) => chip.shared_buffer(),
        }
    }
}

impl<RA> Chip<RA, GpuBackend> for Poseidon2PeripheryChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        match self {
            Self::Register0(chip) => chip.generate_proving_ctx(()),
            Self::Register1(chip) => chip.generate_proving_ctx(()),
        }
    }
}
