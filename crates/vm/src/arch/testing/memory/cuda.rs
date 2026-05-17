use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::memory::air::{MemoryDummyAir, MemoryDummyChip},
        MemoryCellType, MemoryConfig, BLOCK_FE_WIDTH, BUS_PTR_SCALE, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{pack_u8_block_value, MemoryBridge, MemoryBus},
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChipGPU},
    Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    prover::AirProvingContext,
};

use crate::{
    cuda_abi::memory_testing,
    system::cuda::{memory::MemoryInventoryGPU, poseidon2::Poseidon2PeripheryChipGPU},
};

pub struct DeviceMemoryTester {
    pub(crate) chip: FixedSizeMemoryTester,
    pub memory: TracingMemory,
    pub inventory: MemoryInventoryGPU,
    pub hasher_chip: Option<Arc<Poseidon2PeripheryChipGPU>>,

    // Convenience fields, so we don't have to keep unwrapping
    pub config: MemoryConfig,
    pub mem_bus: MemoryBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl DeviceMemoryTester {
    pub fn new(
        memory: TracingMemory,
        mem_bus: MemoryBus,
        mem_config: MemoryConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        let range_bus = range_checker.cpu_chip.as_ref().unwrap().bus();
        let sbox_regs = 1;
        let poseidon2_periphery = Arc::new(Poseidon2PeripheryChipGPU::new(
            1 << 20, // probably enough for our tests
            sbox_regs,
            device_ctx.clone(),
        ));
        let mut inventory = MemoryInventoryGPU::new(
            mem_config.clone(),
            poseidon2_periphery.clone(),
            device_ctx.clone(),
        );
        inventory.set_initial_memory(&memory.data.memory);
        Self {
            chip: FixedSizeMemoryTester::new(mem_bus, device_ctx),
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

    /// See [`crate::arch::testing::MemoryTester::read`] for address semantics.
    pub fn read<const N: usize>(&mut self, addr_space: usize, addr: usize) -> [F; N] {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            let data = self.read_cells(addr_space, addr);
            std::array::from_fn(|i| data[i])
        } else {
            self.read_bytes::<N>(addr_space, addr)
        }
    }

    pub fn write<const N: usize>(&mut self, addr_space: usize, addr: usize, data: [F; N]) {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            self.write_cells(addr_space, addr, std::array::from_fn(|i| data[i]));
        } else {
            self.write_bytes::<N>(addr_space, addr, data);
        }
    }

    fn read_cells(&mut self, addr_space: usize, cell_idx: usize) -> [F; BLOCK_FE_WIDTH] {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        let (t_prev, data) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                self.memory
                    .read::<F, BLOCK_FE_WIDTH>(addr_space as u32, cell_idx as u32)
            },
            MemoryCellType::U16 => {
                let (t_prev, data) = unsafe {
                    self.memory
                        .read::<u16, BLOCK_FE_WIDTH>(addr_space as u32, cell_idx as u32)
                };
                (t_prev, data.map(F::from_u16))
            }
            other => panic!("DeviceMemoryTester::read unsupported cell type {other:?}"),
        };
        let bus_ptr = (cell_idx * BUS_PTR_SCALE) as u32;
        self.chip.receive(addr_space as u32, bus_ptr, &data, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
        data
    }

    fn write_cells(&mut self, addr_space: usize, cell_idx: usize, data: [F; BLOCK_FE_WIDTH]) {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        let (t_prev, data_prev) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                self.memory
                    .write::<F, BLOCK_FE_WIDTH>(addr_space as u32, cell_idx as u32, data)
            },
            MemoryCellType::U16 => {
                let (t_prev, data_prev) = unsafe {
                    self.memory.write::<u16, BLOCK_FE_WIDTH>(
                        addr_space as u32,
                        cell_idx as u32,
                        data.map(|x| {
                            let v = x.as_canonical_u32();
                            assert!(
                                v <= u16::MAX as u32,
                                "DeviceMemoryTester::write got F value {v} outside u16 range",
                            );
                            v as u16
                        }),
                    )
                };
                (t_prev, data_prev.map(F::from_u16))
            }
            other => panic!("DeviceMemoryTester::write unsupported cell type {other:?}"),
        };
        let bus_ptr = (cell_idx * BUS_PTR_SCALE) as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &data_prev, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
    }

    fn read_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize) -> [F; N] {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "DeviceMemoryTester::read_bytes requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes) = unsafe {
            self.memory
                .read_bytes::<N>(addr_space as u32, byte_ptr as u32)
        };
        let data = bytes.map(F::from_u8);
        let packed = pack_u8_block_value(&std::array::from_fn(|i| data[i]));
        self.chip
            .receive(addr_space as u32, byte_ptr as u32, &packed, t_prev);
        self.chip
            .send(addr_space as u32, byte_ptr as u32, &packed, t);
        data
    }

    fn write_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize, data: [F; N]) {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "DeviceMemoryTester::write_bytes requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes_prev) = unsafe {
            self.memory.write_bytes::<N>(
                addr_space as u32,
                byte_ptr as u32,
                data.map(|x| {
                    let v = x.as_canonical_u32();
                    assert!(
                        v <= u8::MAX as u32,
                        "DeviceMemoryTester::write_bytes got F value {v} outside u8 range",
                    );
                    v as u8
                }),
            )
        };
        let data_prev = bytes_prev.map(F::from_u8);
        let packed_prev = pack_u8_block_value(&std::array::from_fn(|i| data_prev[i]));
        let packed_new = pack_u8_block_value(&std::array::from_fn(|i| data[i]));
        self.chip
            .receive(addr_space as u32, byte_ptr as u32, &packed_prev, t_prev);
        self.chip
            .send(addr_space as u32, byte_ptr as u32, &packed_new, t);
    }
}

pub struct FixedSizeMemoryTester(pub(crate) MemoryDummyChip<F>, GpuDeviceCtx);

impl FixedSizeMemoryTester {
    pub fn new(bus: MemoryBus, device_ctx: GpuDeviceCtx) -> Self {
        Self(MemoryDummyChip::new(MemoryDummyAir::new(bus)), device_ctx)
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

impl<RA> Chip<RA, GpuBackend> for FixedSizeMemoryTester {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let width = BaseAir::<F>::width(&self.0.air);
        let height = (self.0.trace.len() / width).next_power_of_two();

        let mut records = self.0.trace.clone();
        records.resize(height * width, F::ZERO);
        let num_records = height;

        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, &self.1);
        trace.buffer().fill_zero_on(&self.1).unwrap();
        unsafe {
            memory_testing::tracegen(
                trace.buffer(),
                height,
                width,
                &records.to_device_on(&self.1).unwrap(),
                num_records,
                BLOCK_FE_WIDTH,
                self.1.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
