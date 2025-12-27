use std::{collections::HashMap, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::memory::air::{MemoryDummyAir, MemoryDummyChip},
        MemoryConfig,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryBus},
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChipGPU};
use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    p3_field::{FieldAlgebra, PrimeField32},
    prover::types::AirProvingContext,
    Chip, ChipUsageGetter,
};

use crate::{
    cuda_abi::memory_testing,
    system::cuda::{memory::MemoryInventoryGPU, poseidon2::Poseidon2PeripheryChipGPU},
};

pub struct DeviceMemoryTester {
    pub chip_for_block: HashMap<usize, FixedSizeMemoryTester>,
    pub memory: TracingMemory,
    pub inventory: MemoryInventoryGPU,
    pub hasher_chip: Option<Arc<Poseidon2PeripheryChipGPU>>,

    // Convenience fields, so we don't have to keep unwrapping
    pub config: MemoryConfig,
    pub mem_bus: MemoryBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl DeviceMemoryTester {
    pub fn volatile(
        memory: TracingMemory,
        mem_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let mut chip_for_block = HashMap::new();
        for log_block_size in 0..6 {
            let block_size = 1 << log_block_size;
            chip_for_block.insert(block_size, FixedSizeMemoryTester::new(mem_bus, block_size));
        }
        let range_bus = range_checker.cpu_chip.as_ref().unwrap().bus();
        Self {
            chip_for_block,
            memory,
            inventory: MemoryInventoryGPU::volatile(mem_config.clone(), range_checker),
            hasher_chip: None,
            config: mem_config,
            mem_bus,
            range_bus,
        }
    }

    pub fn persistent(
        memory: TracingMemory,
        mem_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let mut chip_for_block = HashMap::new();
        for log_block_size in 0..6 {
            let block_size = 1 << log_block_size;
            chip_for_block.insert(block_size, FixedSizeMemoryTester::new(mem_bus, block_size));
        }
        let range_bus = range_checker.cpu_chip.as_ref().unwrap().bus();
        let sbox_regs = 1;
        let poseidon2_periphery = Arc::new(Poseidon2PeripheryChipGPU::new(
            1 << 20, // probably enough for our tests
            sbox_regs,
        ));
        let mut inventory = MemoryInventoryGPU::persistent(
            mem_config.clone(),
            range_checker,
            poseidon2_periphery.clone(),
        );
        inventory.set_initial_memory(&memory.data.memory);
        Self {
            chip_for_block,
            memory,
            inventory,
            hasher_chip: Some(poseidon2_periphery),
            config: mem_config,
            mem_bus,
            range_bus,
        }
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(self.mem_bus, self.config.timestamp_max_bits, self.range_bus)
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

pub struct FixedSizeMemoryTester(pub(crate) MemoryDummyChip<F>);

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
        let height = self.0.current_trace_height().next_power_of_two();
        let width = self.0.trace_width();

        let mut records = self.0.trace.clone();
        records.resize(height * width, F::ZERO);
        let num_records = height;

        let trace = DeviceMatrix::<F>::with_capacity(height, width);
        unsafe {
            memory_testing::tracegen(
                trace.buffer(),
                height,
                width,
                &records.to_device().unwrap(),
                num_records,
                self.0.air.block_size,
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
