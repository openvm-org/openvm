use std::{collections::HashMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use self::{access::NewMemoryAccessCols, dimensions::MemoryDimensions, interface::MemoryInterface};
use super::{audit::MemoryAuditChip, interface::MemoryExpandChip};
use crate::memory::{decompose, OpType};

pub mod access;
pub mod dimensions;
pub mod interface;

#[cfg(test)]
mod tests;

#[derive(Copy, Clone)]
pub struct AccessCell<const WORD_SIZE: usize, F: Field> {
    pub data: [F; WORD_SIZE],
    pub clk: F,
}

pub struct MemoryManager<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    pub interface_chip: MemoryInterface<NUM_WORDS, WORD_SIZE, F>,
    /// Maps (addr_space, pointer) to (data, timestamp)
    // TODO[osama]: this shouldn't always start as empty in the case of continuations
    memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryManager<NUM_WORDS, WORD_SIZE, F>
{
    pub fn with_persistent_memory(
        memory_dimensions: MemoryDimensions,
        memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
    ) -> Self {
        Self {
            interface_chip: MemoryInterface::Persistent(MemoryExpandChip::new(memory_dimensions)),
            memory,
        }
    }

    pub fn with_volatile_memory(
        memory_dimensions: MemoryDimensions,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            interface_chip: MemoryInterface::Volatile(MemoryAuditChip::new(
                memory_dimensions.as_max_bits(),
                memory_dimensions.address_height,
                decomp,
                range_checker,
            )),
            memory: HashMap::new(),
        }
    }

    pub fn read_word(
        &mut self,
        clk: F,
        addr_space: F,
        pointer: F,
    ) -> NewMemoryAccessCols<WORD_SIZE, F> {
        if addr_space == F::zero() {
            let data = decompose(pointer);
            return NewMemoryAccessCols::<WORD_SIZE, F>::new(
                addr_space,
                pointer,
                F::from_canonical_u8(OpType::Read as u8),
                data,
                clk,
                data,
                clk,
            );
        }
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cell = self.memory.get_mut(&(addr_space, pointer)).unwrap();
        let (old_clk, old_data) = (cell.clk, cell.data);
        assert!(old_clk < clk);

        // Updating AccessCell
        cell.clk = clk;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, clk);

        NewMemoryAccessCols::<WORD_SIZE, F>::new(
            addr_space,
            pointer,
            F::from_canonical_u8(OpType::Read as u8),
            old_data,
            old_clk,
            cell.data,
            cell.clk,
        )
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_word(&self, address_space: F, pointer: F) -> [F; WORD_SIZE] {
        self.memory.get(&(address_space, pointer)).unwrap().data
    }

    pub fn write_word(
        &mut self,
        clk: F,
        addr_space: F,
        pointer: F,
        data: [F; WORD_SIZE],
        // TODO[osama]: this should actually return MemoryOfflineCheckerCols
    ) -> NewMemoryAccessCols<WORD_SIZE, F> {
        assert!(addr_space != F::zero());
        println!("writing word pointer: {:?}", pointer.as_canonical_u32());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cell = self
            .memory
            .entry((addr_space, pointer))
            .or_insert(AccessCell {
                data: [F::zero(); WORD_SIZE],
                clk: F::zero(),
            });
        let (old_clk, old_data) = (cell.clk, cell.data);
        assert!(old_clk < clk);

        // Updating AccessCell
        cell.clk = clk;
        cell.data = data;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, old_clk);

        NewMemoryAccessCols::<WORD_SIZE, F>::new(
            addr_space,
            pointer,
            F::from_canonical_u8(OpType::Write as u8),
            old_data,
            old_clk,
            data,
            clk,
        )
    }

    pub fn generate_memory_interface_trace(&self) -> RowMajorMatrix<F> {
        let all_addresses = self.interface_chip.all_addresses();
        let mut final_memory = HashMap::new();
        for (addr_space, pointer) in all_addresses {
            final_memory.insert(
                (addr_space, pointer),
                *self.memory.get(&(addr_space, pointer)).unwrap(),
            );
        }

        self.interface_chip
            .generate_trace(final_memory, (2 * self.memory.len()).next_power_of_two())
    }

    // pub fn memory_clone(&self) -> HashMap<(F, F), F> {
    //     self.memory.clone()
    // }

    // pub fn read_elem(&mut self, timestamp: usize, address_space: F, address: F) -> F {
    //     compose(self.read_word(timestamp, address_space, address))
    // }

    // /// Reads an element directly from memory without updating internal state.
    // ///
    // /// Any value returned is unconstrained.
    // pub fn unsafe_read_elem(&self, address_space: F, address: F) -> F {
    //     compose(self.unsafe_read_word(address_space, address))
    // }

    // pub fn write_elem(&mut self, timestamp: usize, address_space: F, address: F, data: F) {
    //     self.write_word(timestamp, address_space, address, decompose(data));
    // }
}
