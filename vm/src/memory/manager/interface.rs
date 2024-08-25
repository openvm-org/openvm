use std::collections::HashMap;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::AccessCell;
use crate::memory::audit::MemoryAuditChip;

pub enum MemoryInterface<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    Volatile(MemoryAuditChip<WORD_SIZE, F>),
    // Persistent(MemoryExpandInterfaceChip<NUM_WORDS, WORD_SIZE, F>),
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryInterface<NUM_WORDS, WORD_SIZE, F>
{
    pub fn touch_address(&mut self, addr_space: F, pointer: F, data: [F; WORD_SIZE]) {
        match self {
            MemoryInterface::Volatile(ref mut audit_chip) => {
                audit_chip.touch_address(addr_space, pointer, data);
            } // MemoryInterface::Persistent(ref mut expand_chip) => {
              //     expand_chip.touch_address(addr_space, pointer, data, clk);
              // }
        }
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        match self {
            MemoryInterface::Volatile(ref audit_chip) => audit_chip.all_addresses(),
            // MemoryInterface::Persistent(ref expand_chip) => expand_chip.all_addresses(),
        }
    }

    pub fn generate_trace(
        &self,
        final_memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
    ) -> RowMajorMatrix<F> {
        match self {
            MemoryInterface::Volatile(ref audit_chip) => {
                let final_memory_btree = final_memory.into_iter().collect();
                audit_chip.generate_trace(&final_memory_btree)
            } // MemoryInterface::Persistent(ref expand_chip) => {
              //     expand_chip.generate_trace(&final_memory, trace_height)
              // }
        }
    }

    pub fn current_height(&self) -> usize {
        match self {
            MemoryInterface::Volatile(ref audit_chip) => audit_chip.current_height(),
        }
    }
}
