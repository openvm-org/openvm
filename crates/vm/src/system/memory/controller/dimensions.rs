use derive_new::new;
use openvm_stark_backend::p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET},
    system::memory::CHUNK,
};

// indicates that there are 2^`addr_space_height` address spaces numbered starting from 1,
// and that each address space has 2^`address_height` addresses numbered starting from 0
#[derive(Clone, Copy, Debug, Serialize, Deserialize, new)]
pub struct MemoryDimensions {
    /// Address space height
    pub addr_space_height: usize,
    /// Pointer height
    pub address_height: usize,
}

impl MemoryDimensions {
    pub fn overall_height(&self) -> usize {
        self.addr_space_height + self.address_height
    }
    /// Convert an address label (address space, block id) to its index in the memory merkle tree.
    ///
    /// Assumes that `label = (addr_space, block_id)` satisfies `block_id < 2^address_height`.
    ///
    /// This function is primarily for internal use for accessing the memory merkle tree.
    /// Users should use a higher-level API when possible.
    pub fn label_to_index(&self, (addr_space, block_id): (u32, u32)) -> u64 {
        debug_assert!(
            block_id < (1 << self.address_height),
            "block_id={block_id} exceeds address_height={}",
            self.address_height
        );
        (((addr_space - ADDR_SPACE_OFFSET) as u64) << self.address_height) + block_id as u64
    }

    /// Convert an index in the memory merkle tree to an address label (address space, block id).
    ///
    /// This function performs the inverse operation of `label_to_index`.
    pub fn index_to_label(&self, index: u64) -> (u32, u32) {
        let block_id = (index & ((1 << self.address_height) - 1)) as u32;
        let addr_space = (index >> self.address_height) as u32 + ADDR_SPACE_OFFSET;
        (addr_space, block_id)
    }
}

impl MemoryConfig {
    pub fn memory_dimensions(&self) -> MemoryDimensions {
        MemoryDimensions {
            addr_space_height: self.addr_space_height,
            address_height: self.pointer_max_bits - log2_strict_usize(CHUNK),
        }
    }
}
