use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    DEFERRAL_AS, DIGEST_WIDTH, MEMORY_PAGE_BITS,
};

use crate::{
    arch::{SystemConfig, ADDR_SPACE_OFFSET, BOUNDARY_AIR_ID, MERKLE_AIR_ID, U16_CELL_SIZE},
    system::memory::dimensions::MemoryDimensions,
};

/// Upper bound on number of memory pages accessed per instruction. Used for buffer allocation.
pub const MAX_MEM_PAGE_OPS_PER_INSN: usize = 1 << 16;
// Initial allocation only. Correctness is preserved by `Vec::reserve` for large
// range accesses; this avoids preallocating the worst-case per-instruction page
// count for the common scalar-access path.
const INITIAL_CHECKPOINT_PAGE_ACCESSES_PER_INSN: usize = 16;
// Shift amounts from address-space pointer units to memory Merkle leaves.
const BYTE_PTRS_PER_LEAF_BITS: u32 = (U16_CELL_SIZE * DIGEST_WIDTH).ilog2();
const DEFERRAL_PTRS_PER_LEAF_BITS: u32 = DIGEST_WIDTH.ilog2();

// Masks with the first bit set in each aligned group of leaves. These represent
// group occupancy at each page-local Merkle level.
const FIRST_BIT_PER_PAIR: u64 = 0x5555_5555_5555_5555;
const FIRST_BIT_PER_NIBBLE: u64 = 0x1111_1111_1111_1111;
const FIRST_BIT_PER_BYTE: u64 = 0x0101_0101_0101_0101;
const FIRST_BIT_PER_U16: u64 = 0x0001_0001_0001_0001;
const FIRST_BIT_PER_U32: u64 = 0x0000_0001_0000_0001;
const FIRST_BIT_PER_U64: u64 = 0x0000_0000_0000_0001;

#[derive(Clone, Debug)]
pub struct BitSet {
    words: Box<[u64]>,
    dirty_words: Vec<usize>,
}

impl BitSet {
    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0; num_bits.div_ceil(u64::BITS as usize)].into_boxed_slice(),
            dirty_words: Vec::new(),
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
        let old_word = *word;
        let was_set = (old_word & mask) != 0;
        if old_word == 0 {
            self.dirty_words.push(word_index);
        }
        *word = old_word | mask;
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
            let old_word = *word;
            ret += mask_bits - (old_word & mask).count_ones();
            if old_word == 0 {
                self.dirty_words.push(start_word_index);
            }
            *word = old_word | mask;
        } else {
            let end_bit = (end & 63) as u32;
            let mask_bits = 64 - start_bit;
            let mask = u64::MAX << start_bit;
            // SAFETY: Caller ensures start < end and end <= self.words.len() * 64,
            // so start_word_index < self.words.len()
            let start_word = unsafe { self.words.get_unchecked_mut(start_word_index) };
            let old_start_word = *start_word;
            ret += mask_bits - (old_start_word & mask).count_ones();
            if old_start_word == 0 {
                self.dirty_words.push(start_word_index);
            }
            *start_word = old_start_word | mask;

            let mask_bits = end_bit;
            let mask = if end_bit == 0 {
                0
            } else {
                u64::MAX >> (64 - end_bit)
            };
            // SAFETY: Caller ensures end <= self.words.len() * 64, so
            // end_word_index < self.words.len()
            let end_word = unsafe { self.words.get_unchecked_mut(end_word_index) };
            let old_end_word = *end_word;
            ret += mask_bits - (old_end_word & mask).count_ones();
            if mask != 0 && old_end_word == 0 {
                self.dirty_words.push(end_word_index);
            }
            *end_word = old_end_word | mask;
        }

        if start_word_index + 1 < end_word_index {
            for i in (start_word_index + 1)..end_word_index {
                // SAFETY: Caller ensures proper start and end, so i is within bounds
                // of self.words.len()
                let word = unsafe { self.words.get_unchecked_mut(i) };
                let old_word = *word;
                ret += old_word.count_zeros();
                if old_word == 0 {
                    self.dirty_words.push(i);
                }
                *word = u64::MAX;
            }
        }
        ret as usize
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        for word_index in self.dirty_words.drain(..) {
            // SAFETY: dirty_words entries are word indices previously written by this BitSet.
            unsafe {
                *self.words.get_unchecked_mut(word_index) = 0;
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PageAccess {
    /// Index into the page table. The leaf occupancy for this page is stored
    /// in `leaf_mask`.
    pub page_id: u32,
    /// Bit `i` is set when leaf `i` inside this 64-leaf page was touched.
    pub leaf_mask: u64,
}

#[derive(Clone, Debug)]
pub struct MemoryPageTracker {
    page_masks: Box<[u64]>,
    dirty_pages: Vec<usize>,
    upper_nodes: BitSet,
    upper_height: usize,
    merkle_nodes: u32,
    leaves: u32,
}

impl MemoryPageTracker {
    pub fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            page_masks: vec![0; num_pages].into_boxed_slice(),
            dirty_pages: Vec::new(),
            upper_nodes: BitSet::new(num_pages),
            upper_height,
            merkle_nodes: 0,
            leaves: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        for page_id in self.dirty_pages.drain(..) {
            // SAFETY: dirty_pages entries are page indices previously written by this tracker.
            unsafe {
                *self.page_masks.get_unchecked_mut(page_id) = 0;
            }
        }
        self.upper_nodes.clear();
        self.merkle_nodes = 0;
        self.leaves = 0;
    }

    #[inline(always)]
    pub fn insert(&mut self, page_id: usize, leaf_mask: u64) -> (u32, u32) {
        debug_assert!(page_id < self.page_masks.len());
        debug_assert!(leaf_mask != 0);

        let page_mask = unsafe { self.page_masks.get_unchecked_mut(page_id) };
        let old_mask = *page_mask;
        let new_mask = old_mask | leaf_mask;
        if new_mask == old_mask {
            return (0, 0);
        }

        *page_mask = new_mask;
        if old_mask == 0 {
            self.dirty_pages.push(page_id);
        }
        // Newly set bits are the only leaves that can add boundary rows or new
        // page-local Merkle nodes.
        let added_mask = leaf_mask & !old_mask;
        let leaves = added_mask.count_ones();
        let mut merkle_nodes = if leaves == 1 {
            // Single-leaf path: test the aligned 2/4/8/16/32/64-leaf groups
            // containing that leaf.
            local_merkle_nodes_added_leaf(old_mask, added_mask.trailing_zeros())
        } else {
            // Multi-leaf path: walk old and added occupancy up the six
            // page-local levels, then count groups newly occupied by added leaves.
            local_merkle_nodes_delta(old_mask, added_mask)
        };
        self.leaves += leaves;

        if old_mask == 0 {
            merkle_nodes += self.insert_upper_path(page_id);
        }
        self.merkle_nodes += merkle_nodes;
        (leaves, merkle_nodes)
    }

    #[inline(always)]
    fn insert_upper_path(&mut self, page_id: usize) -> u32 {
        // Called only when a page first becomes non-empty in this segment.
        // `upper_nodes` makes nearby pages share already-counted ancestors.
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
fn add_level_delta<const SHIFT: u32, const MASK: u64>(
    old_mask: &mut u64,
    added_mask: &mut u64,
) -> u32 {
    *old_mask = (*old_mask | (*old_mask >> SHIFT)) & MASK;
    *added_mask = (*added_mask | (*added_mask >> SHIFT)) & MASK;
    (*added_mask & !*old_mask).count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta(mut old_mask: u64, mut added_mask: u64) -> u32 {
    debug_assert_ne!(added_mask, 0);
    debug_assert_eq!(old_mask & added_mask, 0);

    let mut nodes = 0;
    nodes += add_level_delta::<1, FIRST_BIT_PER_PAIR>(&mut old_mask, &mut added_mask);
    nodes += add_level_delta::<2, FIRST_BIT_PER_NIBBLE>(&mut old_mask, &mut added_mask);
    nodes += add_level_delta::<4, FIRST_BIT_PER_BYTE>(&mut old_mask, &mut added_mask);
    nodes += add_level_delta::<8, FIRST_BIT_PER_U16>(&mut old_mask, &mut added_mask);
    nodes += add_level_delta::<16, FIRST_BIT_PER_U32>(&mut old_mask, &mut added_mask);
    nodes += add_level_delta::<32, FIRST_BIT_PER_U64>(&mut old_mask, &mut added_mask);

    nodes
}

#[inline(always)]
fn aligned_group_is_empty<const GROUP_SIZE: u32>(old_mask: u64, leaf: u32) -> bool {
    // The ancestor at this level covers the aligned group containing `leaf`.
    // If the old mask has no bit in that group, adding `leaf` creates it.
    let group_start = leaf & !(GROUP_SIZE - 1);
    let group_mask = ((1u64 << GROUP_SIZE) - 1) << group_start;
    old_mask & group_mask == 0
}

#[inline(always)]
fn local_merkle_nodes_added_leaf(old_mask: u64, leaf: u32) -> u32 {
    debug_assert!(leaf < u64::BITS);

    if old_mask == 0 {
        return MEMORY_PAGE_BITS as u32;
    }

    let mut nodes = 0;
    // Group sizes 2..32 are the page-local ancestor levels below the
    // 64-leaf page root. The page root already exists for a non-empty page.
    nodes += u32::from(aligned_group_is_empty::<2>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<4>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<8>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<16>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<32>(old_mask, leaf));
    nodes
}

#[derive(Clone, Debug)]
pub struct MemoryCtx {
    memory_dimensions: MemoryDimensions,
    pub page_tracker: MemoryPageTracker,
    pub page_indices_since_checkpoint: Vec<PageAccess>,
    pub page_indices_since_checkpoint_len: usize,
    page_indices_applied_len: usize,
    pending_leaves: u32,
    pending_merkle_nodes: u32,
}

impl MemoryCtx {
    pub fn new(config: &SystemConfig, segment_check_insns: u64) -> Self {
        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        let upper_height = merkle_height.saturating_sub(MEMORY_PAGE_BITS);
        let checkpoint_capacity = Self::initial_checkpoint_capacity(segment_check_insns);

        Self {
            memory_dimensions,
            page_tracker: MemoryPageTracker::new(upper_height),
            page_indices_since_checkpoint: Vec::with_capacity(checkpoint_capacity),
            page_indices_since_checkpoint_len: 0,
            page_indices_applied_len: 0,
            pending_leaves: 0,
            pending_merkle_nodes: 0,
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
        let leaf_bits = if address_space == DEFERRAL_AS {
            DEFERRAL_PTRS_PER_LEAF_BITS
        } else {
            BYTE_PTRS_PER_LEAF_BITS
        };
        let leaf_label = ptr >> leaf_bits;
        let end_leaf_label = end_ptr >> leaf_bits;
        let num_leaves = end_leaf_label - leaf_label + 1;
        debug_assert!(
            leaf_label < (1 << self.memory_dimensions.address_height),
            "leaf_label={leaf_label} exceeds address_height={}",
            self.memory_dimensions.address_height
        );
        let address_space_offset = (((address_space - ADDR_SPACE_OFFSET) as u64)
            << self.memory_dimensions.address_height) as u32;
        let start_leaf_id = address_space_offset + leaf_label;
        let end_leaf_id = start_leaf_id + num_leaves;
        let start_page_id = start_leaf_id >> MEMORY_PAGE_BITS;

        if num_leaves == 1 {
            self.record_page_access_no_len_update(
                start_page_id,
                1u64 << (start_leaf_id & ((1 << MEMORY_PAGE_BITS) - 1)),
            );
            self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
            return;
        }

        let end_page_id = ((end_leaf_id - 1) >> MEMORY_PAGE_BITS) + 1;
        let num_pages = (end_page_id - start_page_id) as usize;
        assert!(
            num_pages <= MAX_MEM_PAGE_OPS_PER_INSN,
            "more than {MAX_MEM_PAGE_OPS_PER_INSN} memory pages accessed in a single instruction"
        );
        if num_pages > 1 {
            self.page_indices_since_checkpoint.reserve(num_pages);
        }

        for page_id in start_page_id..end_page_id {
            let page_start = page_id << MEMORY_PAGE_BITS;
            let page_end = page_start + (1 << MEMORY_PAGE_BITS);
            let start = start_leaf_id.max(page_start) - page_start;
            let end = end_leaf_id.min(page_end) - page_start;
            let leaf_mask = leaf_mask_range(start, end);
            self.record_page_access_no_len_update(page_id, leaf_mask);
        }
        self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
    }

    #[inline(always)]
    fn record_page_access_no_len_update(&mut self, page_id: u32, leaf_mask: u64) {
        debug_assert!(leaf_mask != 0);
        let len = self.page_indices_since_checkpoint.len();
        if len != 0 {
            // SAFETY: len is non-zero, so len - 1 is in bounds.
            let prev = unsafe {
                self.page_indices_since_checkpoint
                    .get_unchecked_mut(len - 1)
            };
            if prev.page_id == page_id {
                // Consecutive accesses to the same page merge in-place; the
                // tracker later deduplicates non-consecutive repeats.
                prev.leaf_mask |= leaf_mask;
                return;
            }
        }

        if len == self.page_indices_since_checkpoint.capacity() {
            self.page_indices_since_checkpoint.reserve(1);
        }

        // SAFETY: capacity was checked above. PageAccess is Copy and has no drop glue.
        unsafe {
            self.page_indices_since_checkpoint
                .as_mut_ptr()
                .add(len)
                .write(PageAccess { page_id, leaf_mask });
            self.page_indices_since_checkpoint.set_len(len + 1);
        }
    }

    #[cfg(feature = "rvr")]
    #[inline(always)]
    pub(crate) fn apply_page_accesses_with_offset(
        &mut self,
        page_offset: u32,
        accesses: &[PageAccess],
    ) {
        let len = accesses.len();
        let ptr = accesses.as_ptr();
        for i in 0..len {
            // SAFETY: i is bounded by accesses.len().
            let access = unsafe { *ptr.add(i) };
            debug_assert!(access.leaf_mask != 0);
            let page_id = page_offset + access.page_id;
            let (leaves, merkle_nodes) =
                self.page_tracker.insert(page_id as usize, access.leaf_mask);
            self.pending_leaves += leaves;
            self.pending_merkle_nodes += merkle_nodes;
        }
    }

    #[cfg(feature = "rvr")]
    #[inline(always)]
    pub(crate) fn reset_segment_without_replay(&mut self, trace_heights: &mut [u32]) {
        self.page_tracker.clear();
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;

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
    }

    /// Initialize state for a new segment
    #[inline(always)]
    pub(crate) fn initialize_segment(&mut self, trace_heights: &mut [u32]) {
        self.page_tracker.clear();
        self.page_indices_applied_len = 0;
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;

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
        self.apply_height_updates(trace_heights);
    }

    /// Updates the checkpoint with current safe state
    #[inline(always)]
    pub(crate) fn update_checkpoint(&mut self) {
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;
    }

    /// Applies boundary and Merkle height deltas for the page/leaf masks recorded
    /// since the last checkpoint.
    ///
    /// Memory leaves form a sparse Merkle tree. Each page contains the 64 leaves
    /// represented by one `u64` leaf mask:
    ///
    /// ```text
    ///        [root]              height h
    ///        /    \
    ///      ...    ...
    ///      /        \
    ///   [page]    [page]         (h - MEMORY_PAGE_BITS) nodes above each page
    ///   / .. \
    ///  L  ..  L                  MEMORY_PAGE_BITS levels inside a 64-leaf page
    /// ```
    ///
    /// `MemoryPageTracker` counts each newly touched leaf once, each newly
    /// required internal node inside that page once, and each shared ancestor
    /// above the page once across all pages in the segment. Each segment has an
    /// initial and final memory tree, so boundary and Merkle row counts are
    /// doubled.
    ///
    /// - BOUNDARY_AIR: `2 * new_leaves` rows
    /// - MERKLE_AIR:   `2 * new_merkle_nodes` rows
    /// - Poseidon2:    `2 * new_leaves + 2 * new_merkle_nodes` hashes
    ///
    /// The Poseidon2 count is still an upper bound because tracegen may
    /// deduplicate equal hash inputs by value.
    #[inline(always)]
    pub(crate) fn apply_height_updates(&mut self, trace_heights: &mut [u32]) {
        let mut leaves = self.pending_leaves;
        let mut merkle_nodes = self.pending_merkle_nodes;
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;

        for &access in &self.page_indices_since_checkpoint[self.page_indices_applied_len..] {
            let (new_leaves, new_merkle_nodes) = self
                .page_tracker
                .insert(access.page_id as usize, access.leaf_mask);
            leaves += new_leaves;
            merkle_nodes += new_merkle_nodes;
        }
        self.page_indices_applied_len = self.page_indices_since_checkpoint.len();

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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_parent_mask(mut mask: u64) -> u64 {
        let mut parents = 0;
        while mask != 0 {
            let bit = mask.trailing_zeros();
            parents |= 1u64 << (bit >> 1);
            mask &= mask - 1;
        }
        parents
    }

    fn reference_local_merkle_nodes(mask: u64) -> u32 {
        let mut nodes = 0;
        let mut level_mask = mask;
        for _ in 0..MEMORY_PAGE_BITS {
            level_mask = reference_parent_mask(level_mask);
            nodes += level_mask.count_ones();
        }
        nodes
    }

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
        let mut tracker = MemoryPageTracker::new(0);
        tracker.insert(0, 1 << 0);
        assert_eq!(tracker.merkle_nodes, 6);
        tracker.insert(0, 1 << 4);
        assert_eq!(tracker.merkle_nodes, 8);
        tracker.insert(0, 1 << 2);
        assert_eq!(tracker.merkle_nodes, 9);
    }

    #[test]
    fn test_local_merkle_nodes_added_matches_reference_delta() {
        let mut old_mask = 0u64;
        let additions = [
            1u64 << 0,
            1u64 << 1,
            1u64 << 4,
            1u64 << 17,
            1u64 << 31,
            1u64 << 32,
            1u64 << 63,
            0x5555_0000_0000_0000,
            0xaaaa_ffff_0000_0000,
            u64::MAX,
        ];
        for leaf_mask in additions {
            let new_mask = old_mask | leaf_mask;
            if new_mask != old_mask {
                let added_mask = new_mask ^ old_mask;
                let expected =
                    reference_local_merkle_nodes(new_mask) - reference_local_merkle_nodes(old_mask);
                if added_mask.count_ones() == 1 {
                    assert_eq!(
                        local_merkle_nodes_added_leaf(old_mask, added_mask.trailing_zeros()),
                        expected
                    );
                }
                assert_eq!(local_merkle_nodes_delta(old_mask, added_mask), expected);
                old_mask = new_mask;
            }
        }

        old_mask = 1;
        for leaf_mask in additions {
            let new_mask = old_mask | leaf_mask;
            if new_mask != old_mask {
                let added_mask = new_mask ^ old_mask;
                let expected =
                    reference_local_merkle_nodes(new_mask) - reference_local_merkle_nodes(old_mask);
                if added_mask.count_ones() == 1 {
                    assert_eq!(
                        local_merkle_nodes_added_leaf(old_mask, added_mask.trailing_zeros()),
                        expected
                    );
                }
                assert_eq!(local_merkle_nodes_delta(old_mask, added_mask), expected);
                old_mask = new_mask;
            }
        }

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1024 {
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            let old_mask = seed;
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            let new_mask = old_mask | seed;
            if new_mask != old_mask {
                let added_mask = new_mask ^ old_mask;
                assert_eq!(
                    local_merkle_nodes_delta(old_mask, added_mask),
                    reference_local_merkle_nodes(new_mask) - reference_local_merkle_nodes(old_mask)
                );
            }
        }
    }

    #[test]
    fn test_page_mask_duplicate_leaf_does_not_change_counts() {
        let mut tracker = MemoryPageTracker::new(3);
        tracker.insert(0, 1 << 0);
        let leaves = tracker.leaves;
        let nodes = tracker.merkle_nodes;
        tracker.insert(0, 1 << 0);
        assert_eq!(tracker.leaves, leaves);
        assert_eq!(tracker.merkle_nodes, nodes);
    }

    #[test]
    fn test_memory_page_tracker_clear_resets_touched_state() {
        let mut tracker = MemoryPageTracker::new(3);
        tracker.insert(0, 1 << 0);
        tracker.insert(7, 1 << 63);
        assert!(tracker.leaves > 0);
        assert!(tracker.merkle_nodes > 0);

        tracker.clear();
        assert_eq!(tracker.leaves, 0);
        assert_eq!(tracker.merkle_nodes, 0);

        tracker.insert(0, 1 << 0);
        assert_eq!(tracker.leaves, 1);
        assert!(tracker.merkle_nodes > 0);
    }

    #[test]
    fn test_adjacent_pages_share_upper_ancestors() {
        let mut tracker = MemoryPageTracker::new(3);
        tracker.insert(0, 1);
        let first = tracker.merkle_nodes;
        tracker.insert(1, 1);
        assert_eq!(tracker.merkle_nodes - first, MEMORY_PAGE_BITS as u32);
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
        range_ctx.apply_height_updates(&mut range_heights);
        explicit_ctx.apply_height_updates(&mut explicit_heights);
        assert_eq!(range_heights, explicit_heights);
    }
}
