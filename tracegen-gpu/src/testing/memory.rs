use std::{collections::HashMap, slice::from_raw_parts};

use openvm_circuit::{
    arch::testing::memory::air::{MemoryDummyAir, MemoryDummyChip},
    system::memory::{offline_checker::MemoryBus, online::TracingMemory, MemoryController},
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    p3_field::{FieldAlgebra, PrimeField32},
    prover::types::AirProvingContext,
    Chip, ChipUsageGetter,
};
use rand::Rng;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::testing::cuda::memory_testing;

pub struct DeviceMemoryTester {
    pub chip_for_block: HashMap<usize, FixedSizeMemoryTester>,
    pub memory: TracingMemory,
    pub controller: MemoryController<F>,
}

impl DeviceMemoryTester {
    pub fn new(memory: TracingMemory, controller: MemoryController<F>) -> Self {
        let mut chip_for_block = HashMap::new();
        for log_block_size in 0..6 {
            let block_size = 1 << log_block_size;
            chip_for_block.insert(
                block_size,
                FixedSizeMemoryTester::new(controller.memory_bus, block_size),
            );
        }
        Self {
            chip_for_block,
            memory,
            controller,
        }
    }

    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        let t = self.memory.timestamp();
        let (t_prev, data) = if addr_space <= 3 {
            let (t_prev, data) =
                unsafe { self.memory.read::<u8, N, 4>(addr_space as u32, ptr as u32) };
            (t_prev, data.map(F::from_canonical_u8))
        } else {
            unsafe { self.memory.read::<F, N, 1>(addr_space as u32, ptr as u32) }
        };
        self.chip_for_block.get_mut(&N).unwrap().receive(
            addr_space as u32,
            ptr as u32,
            &data,
            t_prev,
        );
        self.chip_for_block
            .get_mut(&N)
            .unwrap()
            .send(addr_space as u32, ptr as u32, &data, t);
        data
    }

    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        let t = self.memory.timestamp();
        let (t_prev, data_prev) = if addr_space <= 3 {
            let (t_prev, data_prev) = unsafe {
                self.memory.write::<u8, N, 4>(
                    addr_space as u32,
                    ptr as u32,
                    data.map(|x| x.as_canonical_u32() as u8),
                )
            };
            (t_prev, data_prev.map(F::from_canonical_u8))
        } else {
            unsafe {
                self.memory
                    .write::<F, N, 1>(addr_space as u32, ptr as u32, data)
            }
        };
        self.chip_for_block.get_mut(&N).unwrap().receive(
            addr_space as u32,
            ptr as u32,
            &data_prev,
            t_prev,
        );
        self.chip_for_block
            .get_mut(&N)
            .unwrap()
            .send(addr_space as u32, ptr as u32, &data, t);
    }
}

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}

pub struct FixedSizeMemoryTester(MemoryDummyChip<F>);

impl FixedSizeMemoryTester {
    pub fn new(bus: MemoryBus, block_size: usize) -> Self {
        Self(MemoryDummyChip::new(MemoryDummyAir::new(bus, block_size)))
    }

    pub fn send(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32) {
        self.0.send(addr_space, ptr, data, timestamp);
    }

    pub fn receive(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32) {
        self.0.receive(addr_space, ptr, data, timestamp);
    }

    pub fn push(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32, count: F) {
        self.0.push(addr_space, ptr, data, timestamp, count);
    }
}

impl ChipUsageGetter for FixedSizeMemoryTester {
    fn air_name(&self) -> String {
        self.0.air_name()
    }

    fn current_trace_height(&self) -> usize {
        self.0.current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.0.trace_width()
    }
}

impl<RA> Chip<RA, GpuBackend> for FixedSizeMemoryTester {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = next_power_of_two_or_zero(self.0.current_trace_height());
        let width = self.0.trace_width();

        if height == 0 {
            return AirProvingContext {
                cached_mains: vec![],
                common_main: None,
                public_values: vec![],
            };
        }
        let trace = DeviceMatrix::<F>::with_capacity(height, width);

        let records = &self.0.trace;
        let num_records = records.len();

        unsafe {
            let bytes_size = num_records * size_of::<F>();
            let records_bytes = from_raw_parts(records.as_ptr() as *const u8, bytes_size);
            let records = records_bytes.to_device().unwrap();
            memory_testing::tracegen(
                trace.buffer(),
                height,
                width,
                &records,
                num_records,
                self.0.air.block_size,
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
