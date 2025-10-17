use openvm_instructions::riscv::{RV32_NUM_REGISTERS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS};

use crate::{arch::SystemConfig, system::memory::dimensions::MemoryDimensions};

#[derive(Clone, Debug)]
pub struct BitSet {
    words: Box<[u64]>,
}

impl BitSet {
    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0; num_bits.div_ceil(u64::BITS as usize)].into_boxed_slice(),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, index: usize) -> bool {
        let word_index = index >> 6;
        let bit_index = index & 63;
        let mask = 1u64 << bit_index;

        debug_assert!(word_index < self.words.len(), "BitSet index out of bounds");

        // SAFETY: word_index is derived from a memory address that is bounds-checked
        //         during memory access. The bitset is sized to accommodate all valid
        //         memory addresses, so word_index is always within bounds.
        let word = unsafe { self.words.get_unchecked_mut(word_index) };
        let was_set = (*word & mask) != 0;
        *word |= mask;
        !was_set
    }

    /// Set all bits within [start, end) to 1, return the number of flipped bits.
    /// Assumes start < end and end <= self.words.len() * 64.
    #[inline(always)]
    pub fn insert_range(&mut self, start: usize, end: usize) -> usize {
        debug_assert!(start < end);
        debug_assert!(end <= self.words.len() * 64, "BitSet range out of bounds");

        let mut ret = 0;
        let start_word_index = start >> 6;
        let end_word_index = (end - 1) >> 6;
        let start_bit = (start & 63) as u32;

        if start_word_index == end_word_index {
            let end_bit = ((end - 1) & 63) as u32 + 1;
            let mask_bits = end_bit - start_bit;
            let mask = (u64::MAX >> (64 - mask_bits)) << start_bit;
            // SAFETY: Caller ensures start < end and end <= self.words.len() * 64,
            // so start_word_index < self.words.len()
            let word = unsafe { self.words.get_unchecked_mut(start_word_index) };
            ret += mask_bits - (*word & mask).count_ones();
            *word |= mask;
        } else {
            let end_bit = (end & 63) as u32;
            let mask_bits = 64 - start_bit;
            let mask = u64::MAX << start_bit;
            // SAFETY: Caller ensures start < end and end <= self.words.len() * 64,
            // so start_word_index < self.words.len()
            let start_word = unsafe { self.words.get_unchecked_mut(start_word_index) };
            ret += mask_bits - (*start_word & mask).count_ones();
            *start_word |= mask;

            let mask_bits = end_bit;
            let mask = if end_bit == 0 {
                0
            } else {
                u64::MAX >> (64 - end_bit)
            };
            // SAFETY: Caller ensures end <= self.words.len() * 64, so
            // end_word_index < self.words.len()
            let end_word = unsafe { self.words.get_unchecked_mut(end_word_index) };
            ret += mask_bits - (*end_word & mask).count_ones();
            *end_word |= mask;
        }

        if start_word_index + 1 < end_word_index {
            for i in (start_word_index + 1)..end_word_index {
                // SAFETY: Caller ensures proper start and end, so i is within bounds
                // of self.words.len()
                let word = unsafe { self.words.get_unchecked_mut(i) };
                ret += word.count_zeros();
                *word = u64::MAX;
            }
        }
        ret as usize
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        // SAFETY: words is valid for self.words.len() elements
        unsafe {
            std::ptr::write_bytes(self.words.as_mut_ptr(), 0, self.words.len());
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryCtx<const PAGE_BITS: usize> {
    pub page_indices: BitSet,
    memory_dimensions: MemoryDimensions,
    min_block_size_bits: Vec<u8>,
    pub boundary_idx: usize,
    pub merkle_tree_index: Option<usize>,
    pub adapter_offset: usize,
    continuations_enabled: bool,
    chunk: u32,
    chunk_bits: u32,
    page_access_count: usize,
    // Note: 32 is the maximum access adapter size.
    addr_space_access_count: Vec<usize>,
}

impl<const PAGE_BITS: usize> MemoryCtx<PAGE_BITS> {
    pub fn new(config: &SystemConfig) -> Self {
        let chunk = config.initial_block_size() as u32;
        let chunk_bits = chunk.ilog2();

        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        Self {
            // Address height already considers `chunk_bits`.
            page_indices: BitSet::new(1 << (merkle_height.saturating_sub(PAGE_BITS))),
            min_block_size_bits: config.memory_config.min_block_size_bits(),
            boundary_idx: config.memory_boundary_air_id(),
            merkle_tree_index: config.memory_merkle_air_id(),
            adapter_offset: config.access_adapter_air_id_offset(),
            chunk,
            chunk_bits,
            memory_dimensions,
            continuations_enabled: config.continuation_enabled,
            page_access_count: 0,
            addr_space_access_count: vec![0; (1 << memory_dimensions.addr_space_height) + 1],
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.page_indices.clear();
    }

    #[inline(always)]
    pub(crate) fn add_register_merkle_heights(&mut self) {
        if self.continuations_enabled {
            self.update_boundary_merkle_heights(
                RV32_REGISTER_AS,
                0,
                (RV32_NUM_REGISTERS * RV32_REGISTER_NUM_LIMBS) as u32,
            );
        }
    }

    /// For each memory access, record the minimal necessary data to update heights of
    /// memory-related chips. The actual height updates happen during segment checks. The
    /// implementation is in `lazy_update_boundary_heights`.
    #[inline(always)]
    pub(crate) fn update_boundary_merkle_heights(
        &mut self,
        address_space: u32,
        ptr: u32,
        size: u32,
    ) {
        debug_assert!((address_space as usize) < self.addr_space_access_count.len());

        let num_blocks = (size + self.chunk - 1) >> self.chunk_bits;
        let start_chunk_id = ptr >> self.chunk_bits;
        let start_block_id = if self.chunk == 1 {
            start_chunk_id
        } else {
            self.memory_dimensions
                .label_to_index((address_space, start_chunk_id)) as u32
        };
        // Because `self.chunk == 1 << self.chunk_bits`
        let end_block_id = start_block_id + num_blocks;
        let start_page_id = start_block_id >> PAGE_BITS;
        let end_page_id = ((end_block_id - 1) >> PAGE_BITS) + 1;

        for page_id in start_page_id..end_page_id {
            if self.page_indices.insert(page_id as usize) {
                self.page_access_count += 1;
                // SAFETY: address_space passed is usually a hardcoded constant or derived from an
                // Instruction where it is bounds checked before passing
                unsafe {
                    *self
                        .addr_space_access_count
                        .get_unchecked_mut(address_space as usize) += 1;
                }
            }
        }
    }

    #[inline(always)]
    pub fn update_adapter_heights(
        &mut self,
        trace_heights: &mut [u32],
        address_space: u32,
        size_bits: u32,
    ) {
        self.update_adapter_heights_batch(trace_heights, address_space, size_bits, 1);
    }

    #[inline(always)]
    pub fn update_adapter_heights_batch(
        &self,
        trace_heights: &mut [u32],
        address_space: u32,
        size_bits: u32,
        num: u32,
    ) {
        debug_assert!((address_space as usize) < self.min_block_size_bits.len());

        // SAFETY: address_space passed is usually a hardcoded constant or derived from an
        // Instruction where it is bounds checked before passing
        let align_bits = unsafe {
            *self
                .min_block_size_bits
                .get_unchecked(address_space as usize)
        };
        debug_assert!(
            align_bits as u32 <= size_bits,
            "align_bits ({}) must be <= size_bits ({})",
            align_bits,
            size_bits
        );

        for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
            let adapter_idx = self.adapter_offset + adapter_bits as usize - 1;
            debug_assert!(adapter_idx < trace_heights.len());
            // SAFETY: trace_heights is initialized taking access adapters into account
            unsafe {
                *trace_heights.get_unchecked_mut(adapter_idx) +=
                    num << (size_bits - adapter_bits + 1);
            }
        }
    }

    /// Resolve all lazy updates of each memory access for memory adapters/poseidon2/merkle chip.
    #[inline(always)]
    pub(crate) fn lazy_update_boundary_heights(&mut self, trace_heights: &mut [u32]) {
        debug_assert!(self.boundary_idx < trace_heights.len());

        // On page fault, assume we add all leaves in a page
        let leaves = (self.page_access_count << PAGE_BITS) as u32;
        // SAFETY: boundary_idx is a compile time constant within bounds
        unsafe {
            *trace_heights.get_unchecked_mut(self.boundary_idx) += leaves;
        }

        if let Some(merkle_tree_idx) = self.merkle_tree_index {
            debug_assert!(merkle_tree_idx < trace_heights.len());
            debug_assert!(trace_heights.len() >= 2);

            let poseidon2_idx = trace_heights.len() - 2;
            // SAFETY: poseidon2_idx is trace_heights.len() - 2, guaranteed to be in bounds
            unsafe {
                *trace_heights.get_unchecked_mut(poseidon2_idx) += leaves * 2;
            }

            let merkle_height = self.memory_dimensions.overall_height();
            let nodes = (((1 << PAGE_BITS) - 1) + (merkle_height - PAGE_BITS)) as u32;
            // SAFETY: merkle_tree_idx is guaranteed to be in bounds
            unsafe {
                *trace_heights.get_unchecked_mut(poseidon2_idx) += nodes * 2;
                *trace_heights.get_unchecked_mut(merkle_tree_idx) += nodes * 2;
            }
        }
        self.page_access_count = 0;

        for address_space in 0..self.addr_space_access_count.len() {
            // SAFETY: address_space is from 0 to len(), guaranteed to be in bounds
            let x = unsafe { *self.addr_space_access_count.get_unchecked(address_space) };
            if x > 0 {
                // After finalize, we'll need to read it in chunk-sized units for the merkle chip
                self.update_adapter_heights_batch(
                    trace_heights,
                    address_space as u32,
                    self.chunk_bits,
                    (x << PAGE_BITS) as u32,
                );
                // SAFETY: address_space is from 0 to len(), guaranteed to be in bounds
                unsafe {
                    *self
                        .addr_space_access_count
                        .get_unchecked_mut(address_space) = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bitset_insert_range() {
        // 513 bits
        let mut bit_set = BitSet::new(8 * 64 + 1);
        let num_flips = bit_set.insert_range(2, 29);
        assert_eq!(num_flips, 27);
        let num_flips = bit_set.insert_range(1, 31);
        assert_eq!(num_flips, 3);

        let num_flips = bit_set.insert_range(32, 65);
        assert_eq!(num_flips, 33);
        let num_flips = bit_set.insert_range(0, 66);
        assert_eq!(num_flips, 3);
        let num_flips = bit_set.insert_range(0, 66);
        assert_eq!(num_flips, 0);

        let num_flips = bit_set.insert_range(256, 320);
        assert_eq!(num_flips, 64);
        let num_flips = bit_set.insert_range(256, 377);
        assert_eq!(num_flips, 57);
        let num_flips = bit_set.insert_range(100, 513);
        assert_eq!(num_flips, 413 - 121);
    }
}
