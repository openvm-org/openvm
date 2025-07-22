use crate::{
    arch::PUBLIC_VALUES_AIR_ID,
    system::memory::{dimensions::MemoryDimensions, CHUNK},
};

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
        let word_index = index / 64;
        let bit_index = index % 64;
        let mask = 1u64 << bit_index;

        let was_set = (self.words[word_index] & mask) != 0;
        self.words[word_index] |= mask;
        !was_set
    }

    /// Set all bits within [start, end) to 1, return the number of flipped bits.
    #[inline(always)]
    pub fn insert_range(&mut self, start: usize, end: usize) -> usize {
        debug_assert!(start < end);
        let mut ret = 0;
        let start_word_index = start / u64::BITS as usize;
        let end_word_index = (end - 1) / u64::BITS as usize;
        let start_bit = start as u32 % u64::BITS;
        if start_word_index == end_word_index {
            let end_bit = (end - 1) as u32 % u64::BITS + 1;
            let mask_bits = end_bit - start_bit;
            let mask = (u64::MAX >> (u64::BITS - mask_bits)) << start_bit;
            ret += mask_bits - (self.words[start_word_index] & mask).count_ones();
            self.words[start_word_index] |= mask;
        } else {
            let end_bit = end as u32 % u64::BITS;
            let mask_bits = u64::BITS - start_bit;
            let mask = u64::MAX << start_bit;
            ret += mask_bits - (self.words[start_word_index] & mask).count_ones();
            self.words[start_word_index] |= mask;
            let mask_bits = end_bit;
            let (mask, _) = u64::MAX.overflowing_shr(u64::BITS - end_bit);
            ret += mask_bits - (self.words[end_word_index] & mask).count_ones();
            self.words[end_word_index] |= mask;
        }
        if start_word_index + 1 < end_word_index {
            for i in (start_word_index + 1)..end_word_index {
                ret += self.words[i].count_zeros();
                self.words[i] = u64::MAX;
            }
        }
        ret as usize
    }

    pub fn clear(&mut self) {
        for item in self.words.iter_mut() {
            *item = 0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryCtx<const PAGE_BITS: usize> {
    pub page_indices: BitSet,
    memory_dimensions: MemoryDimensions,
    as_byte_alignment_bits: Vec<u8>,
    pub boundary_idx: usize,
    pub merkle_tree_index: Option<usize>,
    pub adapter_offset: usize,
    chunk: u32,
    chunk_bits: u32,
    page_access_count: usize,
    // Note: 32 is the maximum access adapter size.
    addr_space_access_count: Vec<usize>,
}

impl<const PAGE_BITS: usize> MemoryCtx<PAGE_BITS> {
    pub fn new(
        has_public_values_chip: bool,
        continuations_enabled: bool,
        as_byte_alignment_bits: Vec<u8>,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        let boundary_idx = if has_public_values_chip {
            PUBLIC_VALUES_AIR_ID + 1
        } else {
            PUBLIC_VALUES_AIR_ID
        };

        let merkle_tree_index = if continuations_enabled {
            Some(boundary_idx + 1)
        } else {
            None
        };

        let adapter_offset = if continuations_enabled {
            boundary_idx + 2
        } else {
            boundary_idx + 1
        };

        let chunk = if continuations_enabled {
            // Persistent memory uses CHUNK-sized blocks
            CHUNK as u32
        } else {
            // Volatile memory uses single units
            1
        };

        let chunk_bits = chunk.ilog2();
        let merkle_height = memory_dimensions.overall_height();

        Self {
            // Address height already considers `chunk_bits`.
            page_indices: BitSet::new(1 << (merkle_height.saturating_sub(PAGE_BITS))),
            as_byte_alignment_bits,
            boundary_idx,
            merkle_tree_index,
            adapter_offset,
            chunk,
            chunk_bits,
            memory_dimensions,
            page_access_count: 0,
            addr_space_access_count: vec![0; (1 << memory_dimensions.addr_space_height) + 1],
        }
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.page_indices.clear();
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
                self.addr_space_access_count[address_space as usize] += 1;
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
        let align_bits = self.as_byte_alignment_bits[address_space as usize];
        debug_assert!(
            align_bits as u32 <= size_bits,
            "align_bits ({}) must be <= size_bits ({})",
            align_bits,
            size_bits
        );
        for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
            let adapter_idx = self.adapter_offset + adapter_bits as usize - 1;
            trace_heights[adapter_idx] += num << (size_bits - adapter_bits + 1);
        }
    }

    /// Resolve all lazy updates of each memory access for memory adapters/poseidon2/merkle chip.
    #[inline(always)]
    pub(crate) fn lazy_update_boundary_heights(&mut self, trace_heights: &mut [u32]) {
        // On page fault, assume we add all leaves in a page
        let leaves = (self.page_access_count << PAGE_BITS) as u32;
        trace_heights[self.boundary_idx] += leaves;

        if let Some(merkle_tree_idx) = self.merkle_tree_index {
            let poseidon2_idx = trace_heights.len() - 2;
            trace_heights[poseidon2_idx] += leaves * 2;

            let merkle_height = self.memory_dimensions.overall_height();
            let nodes = (((1 << PAGE_BITS) - 1) + (merkle_height - PAGE_BITS)) as u32;
            trace_heights[poseidon2_idx] += nodes * 2;
            trace_heights[merkle_tree_idx] += nodes * 2;
        }
        self.page_access_count = 0;
        for address_space in 0..self.addr_space_access_count.len() {
            let x = self.addr_space_access_count[address_space];
            if x > 0 {
                // After finalize, we'll need to read it in chunk-sized units for the merkle chip
                self.update_adapter_heights_batch(
                    trace_heights,
                    address_space as u32,
                    self.chunk_bits,
                    (x << PAGE_BITS) as u32,
                );
                self.addr_space_access_count[address_space] = 0;
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
