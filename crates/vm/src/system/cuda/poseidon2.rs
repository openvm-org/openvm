#[cfg(feature = "metrics")]
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use openvm_circuit::{
    primitives::Chip, system::poseidon2::columns::Poseidon2PeripheryCols,
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::DeviceContext,
};
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};

use crate::cuda_abi::poseidon2;

#[derive(Clone)]
pub struct SharedBuffer<T> {
    pub buffer: Arc<DeviceBuffer<T>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

pub struct Poseidon2ChipGPU<const SBOX_REGISTERS: usize> {
    pub ctx: DeviceContext,
    pub records: Arc<DeviceBuffer<F>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    #[cfg(feature = "metrics")]
    pub(crate) current_trace_height: Arc<AtomicUsize>,
}

impl<const SBOX_REGISTERS: usize> Poseidon2ChipGPU<SBOX_REGISTERS> {
    /// Creates a new Poseidon2 chip with a device buffer of `max_buffer_size` field elements.
    /// Each Poseidon2 record occupies `POSEIDON2_WIDTH` (16) field elements, so the buffer
    /// can hold `max_buffer_size / POSEIDON2_WIDTH` records.
    pub fn new(max_buffer_size: usize, ctx: DeviceContext) -> Self {
        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity_on(1, &ctx));
        idx.fill_zero_on(&ctx).unwrap();
        Self {
            ctx: ctx.clone(),
            records: Arc::new(DeviceBuffer::<F>::with_capacity_on(max_buffer_size, &ctx)),
            idx,
            #[cfg(feature = "metrics")]
            current_trace_height: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        SharedBuffer {
            buffer: self.records.clone(),
            idx: self.idx.clone(),
        }
    }

    pub fn trace_width() -> usize {
        Poseidon2PeripheryCols::<F, SBOX_REGISTERS>::width()
    }
}

impl<RA, const SBOX_REGISTERS: usize> Chip<RA, GpuBackend> for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let mut num_records = self.idx.to_host_on(&self.ctx).unwrap()[0] as usize;
        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let counts = DeviceBuffer::<u32>::with_capacity_on(num_records, &self.ctx);
        unsafe {
            let d_num_records = [num_records].to_device_on(&self.ctx).unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &self.records,
                &counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to get temp bytes");
            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.ctx)
            };
            poseidon2::deduplicate_records(
                &self.records,
                &counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to deduplicate records");
            num_records = *d_num_records.to_host_on(&self.ctx).unwrap().first().unwrap();
        }
        #[cfg(feature = "metrics")]
        self.current_trace_height
            .store(num_records, std::sync::atomic::Ordering::Relaxed);
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, Self::trace_width(), &self.ctx);
        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &self.records,
                &counts,
                num_records,
                SBOX_REGISTERS,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to generate trace");
        }
        // Reset state of this chip.
        self.idx.fill_zero_on(&self.ctx).unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}

pub enum Poseidon2PeripheryChipGPU {
    Register0(Poseidon2ChipGPU<0>),
    Register1(Poseidon2ChipGPU<1>),
}

impl Poseidon2PeripheryChipGPU {
    pub fn new(max_buffer_size: usize, sbox_registers: usize, ctx: DeviceContext) -> Self {
        match sbox_registers {
            0 => Self::Register0(Poseidon2ChipGPU::new(max_buffer_size, ctx)),
            1 => Self::Register1(Poseidon2ChipGPU::new(max_buffer_size, ctx)),
            _ => panic!("Invalid number of sbox registers: {sbox_registers}"),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        match self {
            Self::Register0(chip) => chip.shared_buffer(),
            Self::Register1(chip) => chip.shared_buffer(),
        }
    }

    pub fn ctx(&self) -> &DeviceContext {
        match self {
            Self::Register0(chip) => &chip.ctx,
            Self::Register1(chip) => &chip.ctx,
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
