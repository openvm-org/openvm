use air::{MemoryDummyAir, MemoryDummyChip};
use rand::Rng;

use crate::{
    arch::{VmField, CONST_BLOCK_SIZE},
    system::memory::{online::TracingMemory, MemoryController},
};

pub mod air;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// A dummy testing chip that will add unconstrained messages into the [MemoryBus].
/// Stores a log of raw messages to send/receive to the [MemoryBus].
///
/// It will create a [air::MemoryDummyAir] to add messages to MemoryBus.
pub struct MemoryTester<F: VmField> {
    pub chip: MemoryDummyChip<F>,
    pub memory: TracingMemory,
    pub(super) controller: MemoryController<F>,
}

impl<F: VmField> MemoryTester<F> {
    pub fn new(controller: MemoryController<F>, memory: TracingMemory) -> Self {
        let bus = controller.memory_bus;
        let chip = MemoryDummyChip::new(MemoryDummyAir::new(bus, CONST_BLOCK_SIZE));
        Self {
            chip,
            memory,
            controller,
        }
    }

    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        assert_eq!(
            N, CONST_BLOCK_SIZE,
            "All memory accesses must use CONST_BLOCK_SIZE"
        );
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let (t_prev, data) = unsafe { memory.read::<u8, N>(addr_space as u32, ptr as u32) };
        let data = data.map(F::from_u8);
        self.chip
            .receive(addr_space as u32, ptr as u32, &data, t_prev);
        self.chip.send(addr_space as u32, ptr as u32, &data, t);

        data
    }

    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        assert_eq!(
            N, CONST_BLOCK_SIZE,
            "All memory accesses must use CONST_BLOCK_SIZE"
        );
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let (t_prev, data_prev) = unsafe {
            memory.write::<u8, N>(
                addr_space as u32,
                ptr as u32,
                data.map(|x| x.as_canonical_u32() as u8),
            )
        };
        let data_prev = data_prev.map(F::from_u8);
        self.chip
            .receive(addr_space as u32, ptr as u32, &data_prev, t_prev);
        self.chip.send(addr_space as u32, ptr as u32, &data, t);
    }
}

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.random_range(0..MAX_MEMORY - len) / len * len
}
