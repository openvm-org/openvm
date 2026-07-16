use core::mem::size_of;

use air::{MemoryDummyAir, MemoryDummyChip};
use rand::Rng;

use crate::{
    arch::{MemoryCellType, VmField, BLOCK_FE_WIDTH, NUM_RV64_REGISTERS, U16_CELL_SIZE},
    system::memory::{
        offline_checker::pack_u8_block_value, online::TracingMemory, MemoryController,
    },
};

pub mod air;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// A dummy testing chip that sends/receives unconstrained messages on the [MemoryBus].
/// Cell accesses use `BLOCK_FE_WIDTH`.
pub struct MemoryTester<F: VmField> {
    pub(crate) chip: MemoryDummyChip<F>,
    pub memory: TracingMemory,
    pub(super) controller: MemoryController<F>,
}

impl<F: VmField> MemoryTester<F> {
    pub fn new(controller: MemoryController<F>, memory: TracingMemory) -> Self {
        let chip = MemoryDummyChip::new(MemoryDummyAir::new(controller.memory_bus));
        Self {
            chip,
            memory,
            controller,
        }
    }

    /// Reads one AS-native cell block at `ptr`.
    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        const { assert!(N == BLOCK_FE_WIDTH) };
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        let (t_prev, data) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                memory.read::<F, BLOCK_FE_WIDTH>(addr_space as u32, ptr as u32)
            },
            MemoryCellType::U16 => {
                let (t_prev, data) =
                    unsafe { memory.read::<u16, BLOCK_FE_WIDTH>(addr_space as u32, ptr as u32) };
                (t_prev, data.map(F::from_u16))
            }
            other => panic!("MemoryTester::read unsupported cell type {other:?}"),
        };
        self.chip
            .receive(addr_space as u32, ptr as u32, &data, t_prev);
        self.chip.send(addr_space as u32, ptr as u32, &data, t);
        std::array::from_fn(|i| data[i])
    }

    /// Writes one AS-native cell block at `ptr`.
    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        const { assert!(N == BLOCK_FE_WIDTH) };
        let data: [F; BLOCK_FE_WIDTH] = std::array::from_fn(|i| data[i]);
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        let (t_prev, data_prev) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                memory.write::<F, BLOCK_FE_WIDTH>(addr_space as u32, ptr as u32, data)
            },
            MemoryCellType::U16 => {
                let (t_prev, data_prev) = unsafe {
                    memory.write::<u16, BLOCK_FE_WIDTH>(
                        addr_space as u32,
                        ptr as u32,
                        data.map(|x| {
                            let v = x.as_canonical_u32();
                            assert!(
                                v <= u16::MAX as u32,
                                "MemoryTester::write got F value {v} outside u16 range",
                            );
                            v as u16
                        }),
                    )
                };
                (t_prev, data_prev.map(F::from_u16))
            }
            other => panic!("MemoryTester::write unsupported cell type {other:?}"),
        };
        self.chip
            .receive(addr_space as u32, ptr as u32, &data_prev, t_prev);
        self.chip.send(addr_space as u32, ptr as u32, &data, t);
    }

    pub fn read_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize) -> [F; N] {
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "MemoryTester::read_bytes requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes) = unsafe { memory.read_bytes::<N>(addr_space as u32, byte_ptr as u32) };
        let data = bytes.map(F::from_u8);
        let packed = pack_u8_block_value(&std::array::from_fn(|i| data[i]));
        let ptr = (byte_ptr / U16_CELL_SIZE) as u32;
        self.chip.receive(addr_space as u32, ptr, &packed, t_prev);
        self.chip.send(addr_space as u32, ptr, &packed, t);
        data
    }

    pub fn write_bytes<const N: usize>(
        &mut self,
        addr_space: usize,
        byte_ptr: usize,
        data: [F; N],
    ) {
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "MemoryTester::write_bytes requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes_prev) = unsafe {
            memory.write_bytes::<N>(
                addr_space as u32,
                byte_ptr as u32,
                data.map(|x| {
                    let v = x.as_canonical_u32();
                    assert!(
                        v <= u8::MAX as u32,
                        "MemoryTester::write_bytes got F value {v} outside u8 range",
                    );
                    v as u8
                }),
            )
        };
        let data_prev = bytes_prev.map(F::from_u8);
        let packed_prev = pack_u8_block_value(&std::array::from_fn(|i| data_prev[i]));
        let packed_new = pack_u8_block_value(&std::array::from_fn(|i| data[i]));
        let ptr = (byte_ptr / U16_CELL_SIZE) as u32;
        self.chip
            .receive(addr_space as u32, ptr, &packed_prev, t_prev);
        self.chip.send(addr_space as u32, ptr, &packed_new, t);
    }
}

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.random_range(0..MAX_MEMORY - len) / len * len
}

pub fn gen_register_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    let num_aligned_regions = NUM_RV64_REGISTERS * size_of::<u64>() / len;
    rng.random_range(0..num_aligned_regions) * len
}
