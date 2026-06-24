use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    DEFERRAL_AS,
};

use crate::{
    arch::{SystemConfig, BOUNDARY_AIR_ID, MERKLE_AIR_ID, U16_CELL_SIZE},
    system::memory::{dimensions::MemoryDimensions, DIGEST_WIDTH},
};

/// Number of bits needed to index the 64 leaves represented by one page mask.
pub const PAGE_BITS: usize = 6;

/// Upper bound on number of memory pages accessed per instruction. Used for buffer allocation.
pub const MAX_MEM_PAGE_OPS_PER_INSN: usize = 1 << 16;
const INITIAL_CHECKPOINT_PAGE_ACCESSES_PER_INSN: usize = 16;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PageAccess {
    pub page_id: u32,
    pub leaf_mask: u64,
}

#[derive(Clone, Debug)]
pub struct MemoryPageTracker {
    page_masks: Box<[u64]>,
    upper_nodes: BitSet,
    upper_height: usize,
    merkle_nodes: u32,
    leaves: u32,
}

impl MemoryPageTracker {
    pub fn new(num_pages: usize, upper_height: usize) -> Self {
        Self {
            page_masks: vec![0; num_pages].into_boxed_slice(),
            upper_nodes: BitSet::new(num_pages),
            upper_height,
            merkle_nodes: 0,
            leaves: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.page_masks.as_mut_ptr(), 0, self.page_masks.len());
        }
        self.upper_nodes.clear();
        self.merkle_nodes = 0;
        self.leaves = 0;
    }

    #[inline(always)]
    pub fn insert(&mut self, page_id: usize, leaf_mask: u64) {
        debug_assert!(page_id < self.page_masks.len());
        debug_assert!(leaf_mask != 0);

        let page_mask = unsafe { self.page_masks.get_unchecked_mut(page_id) };
        let old_mask = *page_mask;
        let new_mask = old_mask | leaf_mask;
        if new_mask == old_mask {
            return;
        }

        *page_mask = new_mask;
        self.leaves += (new_mask ^ old_mask).count_ones();
        self.merkle_nodes += local_merkle_nodes(new_mask) - local_merkle_nodes(old_mask);

        if old_mask == 0 {
            self.merkle_nodes += self.insert_upper_path(page_id);
        }
    }

    #[inline(always)]
    fn insert_upper_path(&mut self, page_id: usize) -> u32 {
        let mut count = 0;
        let mut node = (1usize << self.upper_height) + page_id;
        while node > 1 {
            node >>= 1;
            if self.upper_nodes.insert(node) {
                count += 1;
            }
        }
        count
    }
}

#[inline(always)]
fn leaf_mask_range(start: u32, end: u32) -> u64 {
    debug_assert!(start < end);
    debug_assert!(end <= 64);
    let width = end - start;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    mask << start
}

#[inline(always)]
fn parent_mask(mut mask: u64) -> u64 {
    let mut parents = 0;
    while mask != 0 {
        let bit = mask.trailing_zeros();
        parents |= 1u64 << (bit >> 1);
        mask &= mask - 1;
    }
    parents
}

#[inline(always)]
fn local_merkle_nodes(mask: u64) -> u32 {
    if mask == 0 {
        return 0;
    }
    if mask == u64::MAX {
        return ((1usize << PAGE_BITS) - 1) as u32;
    }

    let mut nodes = 0;
    let mut level_mask = mask;
    for _ in 0..PAGE_BITS {
        level_mask = parent_mask(level_mask);
        nodes += level_mask.count_ones();
    }
    nodes
}

#[derive(Clone, Debug)]
pub struct MemoryCtx {
    memory_dimensions: MemoryDimensions,
    pub page_tracker: MemoryPageTracker,
    pub page_indices_since_checkpoint: Vec<PageAccess>,
    pub page_indices_since_checkpoint_len: usize,
}

impl MemoryCtx {
    pub fn new(config: &SystemConfig, segment_check_insns: u64) -> Self {
        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        let num_pages = 1 << (merkle_height.saturating_sub(PAGE_BITS));
        let checkpoint_capacity = Self::initial_checkpoint_capacity(segment_check_insns);

        Self {
            memory_dimensions,
            page_tracker: MemoryPageTracker::new(num_pages, merkle_height - PAGE_BITS),
            page_indices_since_checkpoint: Vec::with_capacity(checkpoint_capacity),
            page_indices_since_checkpoint_len: 0,
        }
    }

    #[inline(always)]
    fn initial_checkpoint_capacity(segment_check_insns: u64) -> usize {
        segment_check_insns as usize * INITIAL_CHECKPOINT_PAGE_ACCESSES_PER_INSN
    }

    #[inline(always)]
    pub(crate) fn add_register_merkle_heights(&mut self) {
        self.update_boundary_merkle_heights(
            RV64_REGISTER_AS,
            0,
            (RV64_NUM_REGISTERS * RV64_REGISTER_NUM_LIMBS) as u32,
        );
    }

    /// Records the memory-tree pages touched by `[ptr, ptr + size)`.
    /// For metered callbacks, DEFERRAL_AS ranges are AS-native F-cell ranges and
    /// u16-celled address space ranges are byte ranges.
    #[inline(always)]
    pub(crate) fn update_boundary_merkle_heights(
        &mut self,
        address_space: u32,
        ptr: u32,
        size: u32,
    ) {
        let end_ptr = ptr + size - 1;
        let ptrs_per_leaf = if address_space == DEFERRAL_AS {
            DIGEST_WIDTH as u32
        } else {
            (U16_CELL_SIZE * DIGEST_WIDTH) as u32
        };
        let leaf_bits = ptrs_per_leaf.ilog2();
        let leaf_label = ptr >> leaf_bits;
        let end_leaf_label = end_ptr >> leaf_bits;
        let num_leaves = end_leaf_label - leaf_label + 1;
        let start_leaf_id = self
            .memory_dimensions
            .label_to_index((address_space, leaf_label)) as u32;
        let end_leaf_id = start_leaf_id + num_leaves;
        let start_page_id = start_leaf_id >> PAGE_BITS;
        let end_page_id = ((end_leaf_id - 1) >> PAGE_BITS) + 1;
        let num_pages = (end_page_id - start_page_id) as usize;
        assert!(
            num_pages <= MAX_MEM_PAGE_OPS_PER_INSN,
            "more than {MAX_MEM_PAGE_OPS_PER_INSN} memory pages accessed in a single instruction"
        );
        self.page_indices_since_checkpoint.reserve(num_pages);

        for page_id in start_page_id..end_page_id {
            let page_start = page_id << PAGE_BITS;
            let page_end = page_start + (1 << PAGE_BITS);
            let start = start_leaf_id.max(page_start) - page_start;
            let end = end_leaf_id.min(page_end) - page_start;
            let leaf_mask = leaf_mask_range(start, end);
            self.record_page_access(page_id, leaf_mask);
        }
    }

    #[inline(always)]
    pub(crate) fn record_page_access(&mut self, page_id: u32, leaf_mask: u64) {
        debug_assert!(leaf_mask != 0);
        self.page_indices_since_checkpoint
            .push(PageAccess { page_id, leaf_mask });
        self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
    }

    /// Initialize state for a new segment
    #[inline(always)]
    pub(crate) fn initialize_segment(&mut self, trace_heights: &mut [u32]) {
        self.page_tracker.clear();

        // Reset trace heights for memory chips as 0
        // SAFETY: BOUNDARY_AIR_ID and MERKLE_AIR_ID are compile-time constants within bounds
        unsafe {
            *trace_heights.get_unchecked_mut(BOUNDARY_AIR_ID) = 0;
            *trace_heights.get_unchecked_mut(MERKLE_AIR_ID) = 0;
        }
        let poseidon2_idx = trace_heights.len() - 2;
        // SAFETY: poseidon2_idx is trace_heights.len() - 2, guaranteed to be in bounds
        unsafe {
            *trace_heights.get_unchecked_mut(poseidon2_idx) = 0;
        }

        self.apply_height_updates(trace_heights);

        // Add merkle height contributions for all registers
        self.add_register_merkle_heights();
        self.lazy_update_boundary_heights(trace_heights);
    }

    /// Updates the checkpoint with current safe state
    #[inline(always)]
    pub(crate) fn update_checkpoint(&mut self) {
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
    }

    /// Applies exact boundary and Merkle height deltas for the page/leaf masks
    /// recorded since the last checkpoint. Poseidon2 remains a safe upper bound:
    /// tracegen may deduplicate equal hash inputs by value.
    #[inline(always)]
    fn apply_height_updates(&mut self, trace_heights: &mut [u32]) {
        let old_leaves = self.page_tracker.leaves;
        let old_merkle_nodes = self.page_tracker.merkle_nodes;
        for &access in &self.page_indices_since_checkpoint {
            self.page_tracker
                .insert(access.page_id as usize, access.leaf_mask);
        }

        let leaves = self.page_tracker.leaves - old_leaves;
        let merkle_nodes = self.page_tracker.merkle_nodes - old_merkle_nodes;
        debug_assert!(trace_heights.len() >= 2);
        let poseidon2_idx = trace_heights.len() - 2;
        // SAFETY: BOUNDARY_AIR_ID, MERKLE_AIR_ID, and poseidon2_idx are all within bounds
        unsafe {
            *trace_heights.get_unchecked_mut(BOUNDARY_AIR_ID) += leaves * 2;
            // Poseidon2: 2 hashes per leaf (compression) + 2 per internal node (init + final tree)
            *trace_heights.get_unchecked_mut(poseidon2_idx) += leaves * 2 + merkle_nodes * 2;
            // Merkle AIR: 2 rows per internal node (init + final tree)
            *trace_heights.get_unchecked_mut(MERKLE_AIR_ID) += merkle_nodes * 2;
        }
    }

    /// Resolve all lazy updates of each memory access for poseidon2/merkle chips.
    #[inline(always)]
    pub(crate) fn lazy_update_boundary_heights(&mut self, trace_heights: &mut [u32]) {
        self.apply_height_updates(trace_heights);
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

    #[test]
    fn test_local_merkle_nodes_doc_example() {
        let mut tracker = MemoryPageTracker::new(1, 0);
        tracker.insert(0, 1 << 0);
        assert_eq!(tracker.merkle_nodes, 6);
        tracker.insert(0, 1 << 4);
        assert_eq!(tracker.merkle_nodes, 8);
        tracker.insert(0, 1 << 2);
        assert_eq!(tracker.merkle_nodes, 9);
    }

    #[test]
    fn test_page_mask_duplicate_leaf_does_not_change_counts() {
        let mut tracker = MemoryPageTracker::new(8, 3);
        tracker.insert(0, 1 << 0);
        let leaves = tracker.leaves;
        let nodes = tracker.merkle_nodes;
        tracker.insert(0, 1 << 0);
        assert_eq!(tracker.leaves, leaves);
        assert_eq!(tracker.merkle_nodes, nodes);
    }

    #[test]
    fn test_adjacent_pages_share_upper_ancestors() {
        let mut tracker = MemoryPageTracker::new(8, 3);
        tracker.insert(0, 1);
        let first = tracker.merkle_nodes;
        tracker.insert(1, 1);
        assert_eq!(tracker.merkle_nodes - first, PAGE_BITS as u32);
    }

    #[test]
    fn test_range_insertion_matches_explicit_leaves() {
        let system_config = crate::utils::test_system_config();
        let mut range_ctx = MemoryCtx::new(&system_config, 1);
        let mut explicit_ctx = MemoryCtx::new(&system_config, 1);

        range_ctx.update_boundary_merkle_heights(2, 0, 17);
        for ptr in [0, 16] {
            explicit_ctx.update_boundary_merkle_heights(2, ptr, 1);
        }

        let mut range_heights = vec![0; 6];
        let mut explicit_heights = vec![0; 6];
        range_ctx.lazy_update_boundary_heights(&mut range_heights);
        explicit_ctx.lazy_update_boundary_heights(&mut explicit_heights);
        assert_eq!(range_heights, explicit_heights);
    }
}
