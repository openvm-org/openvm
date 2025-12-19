use std::{collections::HashMap, convert::TryInto};

use air::{MemoryDummyAir, MemoryDummyChip};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::Rng;

use crate::system::memory::{online::TracingMemory, MemoryController};

pub mod air;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// A dummy testing chip that will add unconstrained messages into the [MemoryBus].
/// Stores a log of raw messages to send/receive to the [MemoryBus].
///
/// It will create a [air::MemoryDummyAir] to add messages to MemoryBus.
pub struct MemoryTester<F: Field> {
    /// Map from `block_size` to [MemoryDummyChip] of that block size
    pub chip_for_block: HashMap<usize, MemoryDummyChip<F>>,
    pub memory: TracingMemory,
    pub(super) controller: MemoryController<F>,
}

impl<F: PrimeField32> MemoryTester<F> {
    pub fn new(controller: MemoryController<F>, memory: TracingMemory) -> Self {
        let bus = controller.memory_bus;
        let mut chip_for_block = HashMap::new();
        for log_block_size in 0..6 {
            let block_size = 1 << log_block_size;
            let chip = MemoryDummyChip::new(MemoryDummyAir::new(bus, block_size));
            chip_for_block.insert(block_size, chip);
        }
        Self {
            chip_for_block,
            memory,
            controller,
        }
    }

    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        let memory = &mut self.memory;
        // For AS 1-3, force CONST_BLOCK_SIZE (4-byte) accesses to avoid adapters.
        if addr_space <= 3 && N > 4 {
            let mut out = [F::ZERO; N];
            for (chunk_idx, chunk) in out.chunks_exact_mut(4).enumerate() {
                let chunk_ptr = ptr + chunk_idx * 4;
                let t = memory.timestamp();
                let (t_prev, data) =
                    unsafe { memory.read::<u8, 4, 4>(addr_space as u32, chunk_ptr as u32) };
                let data_f = data.map(F::from_canonical_u8);
                chunk.copy_from_slice(&data_f);
                self.chip_for_block.get_mut(&4).unwrap().receive(
                    addr_space as u32,
                    chunk_ptr as u32,
                    &data_f,
                    t_prev,
                );
                self.chip_for_block.get_mut(&4).unwrap().send(
                    addr_space as u32,
                    chunk_ptr as u32,
                    &data_f,
                    t,
                );
            }
            return out;
        }

        let t = memory.timestamp();
        // TODO: this could be improved if we added a TracingMemory::get_f function
        let (t_prev, data) = if addr_space <= 3 {
            let (t_prev, data) = unsafe { memory.read::<u8, N, 4>(addr_space as u32, ptr as u32) };
            (t_prev, data.map(F::from_canonical_u8))
        } else {
            unsafe { memory.read::<F, N, 1>(addr_space as u32, ptr as u32) }
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
        let memory = &mut self.memory;
        // For AS 1-3, force CONST_BLOCK_SIZE (4-byte) accesses to avoid adapters.
        if addr_space <= 3 && N > 4 {
            for (chunk_idx, chunk) in data.chunks_exact(4).enumerate() {
                let chunk_ptr = ptr + chunk_idx * 4;
                let t = memory.timestamp();
                let chunk_u8: [u8; 4] = chunk
                    .iter()
                    .map(|x| x.as_canonical_u32() as u8)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let (t_prev, data_prev) = unsafe {
                    memory.write::<u8, 4, 4>(addr_space as u32, chunk_ptr as u32, chunk_u8)
                };
                let data_prev_f = data_prev.map(F::from_canonical_u8);
                let chunk_f: [F; 4] = chunk.try_into().unwrap();
                self.chip_for_block.get_mut(&4).unwrap().receive(
                    addr_space as u32,
                    chunk_ptr as u32,
                    &data_prev_f,
                    t_prev,
                );
                self.chip_for_block.get_mut(&4).unwrap().send(
                    addr_space as u32,
                    chunk_ptr as u32,
                    &chunk_f,
                    t,
                );
            }
            return;
        }

        let t = memory.timestamp();
        // TODO: this could be improved if we added a TracingMemory::write_f function
        let (t_prev, data_prev) = if addr_space <= 3 {
            let (t_prev, data_prev) = unsafe {
                memory.write::<u8, N, 4>(
                    addr_space as u32,
                    ptr as u32,
                    data.map(|x| x.as_canonical_u32() as u8),
                )
            };
            (t_prev, data_prev.map(F::from_canonical_u8))
        } else {
            unsafe { memory.write::<F, N, 1>(addr_space as u32, ptr as u32, data) }
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
