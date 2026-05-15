use air::{MemoryDummyAir, MemoryDummyChip};
use openvm_instructions::DEFERRAL_AS;
use rand::Rng;

use crate::{
    arch::{VmField, DEFAULT_BLOCK_SIZE},
    system::memory::{online::TracingMemory, MemoryController},
};

pub mod air;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// A dummy testing chip that sends/receives unconstrained messages on the [MemoryBus].
/// All memory accesses use `DEFAULT_BLOCK_SIZE`.
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

    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        const { assert!(N == DEFAULT_BLOCK_SIZE) };
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let (t_prev, data) = if addr_space as u32 == DEFERRAL_AS {
            unsafe { memory.read::<F, N>(addr_space as u32, ptr as u32) }
        } else {
            let (t_prev, data) = unsafe { memory.read::<u8, N>(addr_space as u32, ptr as u32) };
            (t_prev, data.map(F::from_u8))
        };
        self.chip
            .receive(addr_space as u32, ptr as u32, &data, t_prev);
        self.chip.send(addr_space as u32, ptr as u32, &data, t);

        data
    }

    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        const { assert!(N == DEFAULT_BLOCK_SIZE) };
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let (t_prev, data_prev) = if addr_space as u32 == DEFERRAL_AS {
            unsafe { memory.write::<F, N>(addr_space as u32, ptr as u32, data) }
        } else {
            let (t_prev, data_prev) = unsafe {
                memory.write::<u8, N>(
                    addr_space as u32,
                    ptr as u32,
                    data.map(|x| x.as_canonical_u32() as u8),
                )
            };
            (t_prev, data_prev.map(F::from_u8))
        };
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
