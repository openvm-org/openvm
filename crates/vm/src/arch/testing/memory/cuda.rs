use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::memory::air::{MemoryDummyAir, MemoryDummyChip},
        MemoryCellType, MemoryConfig, BLOCK_FE_WIDTH, BUS_PTR_SCALE, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryBus},
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

    /// See [`crate::arch::testing::MemoryTester::read`] for the dual-N
    /// semantics: `N = BLOCK_FE_WIDTH` is cell-indexed access (`ptr` is a
    /// cell index, chip records `BUS_PTR_SCALE * ptr` as bus pointer);
    /// `N = MEMORY_BLOCK_BYTES` is byte-view against u16 ASes only (`ptr` is
    /// the byte/bus pointer, bytes are packed pairwise into `BLOCK_FE_WIDTH`
    /// field elements for the chip's bus message).
    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            self.read_cells::<N>(addr_space, ptr)
        } else {
            self.read_bytes::<N>(addr_space, ptr)
        }
    }

    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            self.write_cells::<N>(addr_space, ptr, data);
        } else {
            self.write_bytes::<N>(addr_space, ptr, data);
        }
    }

    fn read_cells<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        let (t_prev, data) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                self.memory.read::<F, N>(addr_space as u32, ptr as u32)
            },
            MemoryCellType::U16 => {
                let (t_prev, data) =
                    unsafe { self.memory.read::<u16, N>(addr_space as u32, ptr as u32) };
                (t_prev, data.map(F::from_u16))
            }
            other => panic!("DeviceMemoryTester::read unsupported cell type {other:?}"),
        };
        let bus_ptr = (ptr * BUS_PTR_SCALE) as u32;
        self.chip.receive(addr_space as u32, bus_ptr, &data, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
        data
    }

    fn write_cells<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        let (t_prev, data_prev) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                self.memory
                    .write::<F, N>(addr_space as u32, ptr as u32, data)
            },
            MemoryCellType::U16 => {
                let (t_prev, data_prev) = unsafe {
                    self.memory.write::<u16, N>(
                        addr_space as u32,
                        ptr as u32,
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
        let bus_ptr = (ptr * BUS_PTR_SCALE) as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &data_prev, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
    }

    fn read_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize) -> [F; N] {
        assert_eq!(N, MEMORY_BLOCK_BYTES);
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "DeviceMemoryTester::read byte-view requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes) = unsafe {
            self.memory
                .read::<u8, N>(addr_space as u32, byte_ptr as u32)
        };
        let data = bytes.map(F::from_u8);
        let packed = pack_bytes_for_bus(&data);
        let bus_ptr = byte_ptr as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &packed, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &packed, t);
        data
    }

    fn write_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize, data: [F; N]) {
        assert_eq!(N, MEMORY_BLOCK_BYTES);
        let t = self.memory.timestamp();
        let cell_layout = self.memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "DeviceMemoryTester::write byte-view requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes_prev) = unsafe {
            self.memory.write::<u8, N>(
                addr_space as u32,
                byte_ptr as u32,
                data.map(|x| {
                    let v = x.as_canonical_u32();
                    assert!(
                        v <= u8::MAX as u32,
                        "DeviceMemoryTester::write byte-view got F value {v} outside u8 range",
                    );
                    v as u8
                }),
            )
        };
        let data_prev = bytes_prev.map(F::from_u8);
        let packed_prev = pack_bytes_for_bus(&data_prev);
        let packed_new = pack_bytes_for_bus(&data);
        let bus_ptr = byte_ptr as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &packed_prev, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &packed_new, t);
    }
}

/// Packs `MEMORY_BLOCK_BYTES` u8-typed F values into `BLOCK_FE_WIDTH` packed F
/// values via base-256; matches `bridge::pack_for_bus` and the CPU
/// `MemoryTester::pack_bytes_for_bus` so the chip's bus message exactly equals
/// what a real chip would emit through the legacy bridge pack.
fn pack_bytes_for_bus(data: &[F]) -> [F; BLOCK_FE_WIDTH] {
    assert_eq!(data.len(), MEMORY_BLOCK_BYTES);
    let ratio = MEMORY_BLOCK_BYTES / BLOCK_FE_WIDTH;
    let mut packed = [F::ZERO; BLOCK_FE_WIDTH];
    for i in 0..BLOCK_FE_WIDTH {
        let mut acc = F::ZERO;
        let mut mult = 1u64;
        for k in 0..ratio {
            acc += data[i * ratio + k] * F::from_u64(mult);
            mult *= 256;
        }
        packed[i] = acc;
    }
    packed
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
