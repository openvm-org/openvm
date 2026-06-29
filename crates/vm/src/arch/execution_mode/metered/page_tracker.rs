use openvm_instructions::MEMORY_PAGE_BITS;

// Masks with the first bit set in each aligned group of leaves. These represent
// group occupancy at each page-local Merkle level.
const FIRST_BIT_PER_PAIR: u64 = 0x5555_5555_5555_5555;
const FIRST_BIT_PER_NIBBLE: u64 = 0x1111_1111_1111_1111;
const FIRST_BIT_PER_BYTE: u64 = 0x0101_0101_0101_0101;
const FIRST_BIT_PER_U16: u64 = 0x0001_0001_0001_0001;
const FIRST_BIT_PER_U32: u64 = 0x0000_0001_0000_0001;
const FIRST_BIT_PER_U64: u64 = 0x0000_0000_0000_0001;

/// Old-side leaves and Merkle nodes that were still canonical default values
/// when the current segment started.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DefaultOldCounts {
    pub(super) leaves: u32,
    pub(super) merkle_nodes: u32,
    merkle_node_levels: u64,
}

impl DefaultOldCounts {
    #[inline(always)]
    pub(super) fn add(&mut self, other: Self) {
        self.leaves += other.leaves;
        self.merkle_nodes += other.merkle_nodes;
        self.merkle_node_levels |= other.merkle_node_levels;
    }

    #[inline(always)]
    pub(super) fn estimated_poseidon_rows(self) -> u32 {
        u32::from(self.leaves != 0) + self.merkle_node_levels.count_ones()
    }
}

#[derive(Clone, Debug)]
struct BitSet {
    words: Box<[u64]>,
    dirty_words: Vec<usize>,
}

impl BitSet {
    fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0; num_bits.div_ceil(u64::BITS as usize)].into_boxed_slice(),
            dirty_words: Vec::new(),
        }
    }

    #[inline(always)]
    fn insert(&mut self, index: usize) -> bool {
        // Segment-local sets use dirty-word tracking so `clear` only touches
        // words written in the segment.
        self.insert_impl::<true>(index)
    }

    #[inline(always)]
    fn insert_persistent(&mut self, index: usize) -> bool {
        // Committed occupancy is monotonic and persists across checkpoints.
        self.insert_impl::<false>(index)
    }

    #[inline(always)]
    fn insert_impl<const TRACK_DIRTY: bool>(&mut self, index: usize) -> bool {
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
        if was_set {
            return false;
        }
        if TRACK_DIRTY && old_word == 0 {
            self.dirty_words.push(word_index);
        }
        *word = old_word | mask;
        true
    }

    #[inline(always)]
    fn contains(&self, index: usize) -> bool {
        let word_index = index >> 6;
        let bit_index = index & 63;
        let mask = 1u64 << bit_index;

        debug_assert!(word_index < self.words.len(), "BitSet index out of bounds");

        // SAFETY: word_index is derived from a valid memory tree node index.
        unsafe { (*self.words.get_unchecked(word_index) & mask) != 0 }
    }

    #[inline(always)]
    fn clear(&mut self) {
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MemoryHeightDelta {
    pub(super) leaves: u32,
    pub(super) merkle_nodes: u32,
    pub(super) added_mask: u64,
    pub(super) default_old: DefaultOldCounts,
}

/// Segment-local page accounting.
///
/// This tracker is cleared whenever a new segment starts. It answers: "which
/// leaves and Merkle nodes have already been charged in this segment?"
/// `MemoryOccupancyTracker` is passed into `insert` only to classify old-side
/// nodes as default or non-default.
///
/// Memory leaves form one sparse Merkle tree. Each page contains the 64 leaves
/// represented by one `u64` leaf mask:
///
/// ```text
///        [root]              height h
///        /    \
///      ...    ...
///      /        \
///   [page]    [page]         h - MEMORY_PAGE_BITS nodes above each page
///   / .. \
///  L  ..  L                  MEMORY_PAGE_BITS levels inside a 64-leaf page
/// ```
///
/// The tracker counts each newly touched leaf once, each newly required
/// page-local internal node once, and each upper-tree ancestor once across all
/// pages in the segment.
#[derive(Clone, Debug)]
pub(super) struct MemoryPageTracker {
    page_masks: Box<[u64]>,
    dirty_pages: Vec<usize>,
    upper_nodes: BitSet,
    upper_height: usize,
}

impl MemoryPageTracker {
    pub(super) fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            page_masks: vec![0; num_pages].into_boxed_slice(),
            dirty_pages: Vec::new(),
            upper_nodes: BitSet::new(num_pages),
            upper_height,
        }
    }

    #[inline(always)]
    pub(super) fn clear(&mut self) {
        for page_id in self.dirty_pages.drain(..) {
            // SAFETY: dirty_pages entries are page indices previously written by this tracker.
            unsafe {
                *self.page_masks.get_unchecked_mut(page_id) = 0;
            }
        }
        self.upper_nodes.clear();
    }

    #[inline(always)]
    pub(super) fn insert(
        &mut self,
        page_id: usize,
        leaf_mask: u64,
        occupancy_tracker: &MemoryOccupancyTracker,
    ) -> MemoryHeightDelta {
        debug_assert!(page_id < self.page_masks.len());
        debug_assert!(leaf_mask != 0);

        let page_mask = unsafe { self.page_masks.get_unchecked_mut(page_id) };
        let old_mask = *page_mask;
        let new_mask = old_mask | leaf_mask;
        if new_mask == old_mask {
            return MemoryHeightDelta::default();
        }

        *page_mask = new_mask;
        if old_mask == 0 {
            self.dirty_pages.push(page_id);
        }

        let added_mask = leaf_mask & !old_mask;
        let committed_mask = occupancy_tracker.page_mask(page_id);
        let default_leaf_mask = added_mask & !committed_mask;
        let single_added_leaf = added_mask.is_power_of_two();
        let leaves = if single_added_leaf {
            1
        } else {
            added_mask.count_ones()
        };

        let (mut merkle_nodes, mut default_old) = if default_leaf_mask == 0 {
            (
                if single_added_leaf {
                    local_merkle_nodes_added_leaf(old_mask, added_mask.trailing_zeros())
                } else {
                    local_merkle_nodes_delta(old_mask, added_mask)
                },
                DefaultOldCounts::default(),
            )
        } else if single_added_leaf {
            local_merkle_nodes_added_leaf_with_default(
                old_mask,
                added_mask.trailing_zeros(),
                committed_mask,
            )
        } else {
            local_merkle_nodes_delta_with_default(old_mask, added_mask, committed_mask)
        };
        if default_leaf_mask != 0 {
            default_old.leaves = if single_added_leaf {
                1
            } else {
                default_leaf_mask.count_ones()
            };
        }

        if old_mask == 0 {
            if committed_mask == 0 {
                let (upper_nodes, upper_default_old) =
                    self.insert_upper_path(page_id, occupancy_tracker);
                merkle_nodes += upper_nodes;
                default_old.add(upper_default_old);
            } else {
                merkle_nodes += self.insert_upper_path_count_only(page_id);
            }
        }
        MemoryHeightDelta {
            leaves,
            merkle_nodes,
            added_mask,
            default_old,
        }
    }

    #[inline(always)]
    fn insert_upper_path(
        &mut self,
        page_id: usize,
        occupancy_tracker: &MemoryOccupancyTracker,
    ) -> (u32, DefaultOldCounts) {
        let mut count = 0;
        let mut default_old = DefaultOldCounts::default();
        let mut node = (1usize << self.upper_height) + page_id;
        let mut height = MEMORY_PAGE_BITS + 1;
        while node > 1 {
            node >>= 1;
            if self.upper_nodes.insert(node) {
                count += 1;
                if !occupancy_tracker.upper_contains(node) {
                    default_old.merkle_nodes += 1;
                    default_old.merkle_node_levels |= 1u64 << height;
                }
            } else {
                break;
            }
            height += 1;
        }
        (count, default_old)
    }

    #[inline(always)]
    fn insert_upper_path_count_only(&mut self, page_id: usize) -> u32 {
        let mut count = 0;
        let mut node = (1usize << self.upper_height) + page_id;
        while node > 1 {
            node >>= 1;
            if self.upper_nodes.insert(node) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }
}

/// Occupancy committed at the last safe checkpoint.
///
/// This tracker persists across segment-local replays and is advanced only by
/// `MemoryCtx::update_checkpoint`. It is seeded from nonzero initial memory, so
/// pages or ancestors missing here are known canonical default old values.
#[derive(Clone, Debug)]
pub(super) struct MemoryOccupancyTracker {
    page_masks: Box<[u64]>,
    upper_nodes: BitSet,
    upper_height: usize,
}

impl MemoryOccupancyTracker {
    pub(super) fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            page_masks: vec![0; num_pages].into_boxed_slice(),
            upper_nodes: BitSet::new(num_pages),
            upper_height,
        }
    }

    #[inline(always)]
    pub(super) fn page_mask(&self, page_id: usize) -> u64 {
        debug_assert!(page_id < self.page_masks.len());
        unsafe { *self.page_masks.get_unchecked(page_id) }
    }

    #[inline(always)]
    fn upper_contains(&self, node: usize) -> bool {
        self.upper_nodes.contains(node)
    }

    #[inline(always)]
    pub(super) fn mark_existing_page(&mut self, page_id: usize, leaf_mask: u64) {
        debug_assert!(page_id < self.page_masks.len());
        debug_assert!(leaf_mask != 0);

        let page_mask = unsafe { self.page_masks.get_unchecked_mut(page_id) };
        let old_mask = *page_mask;
        *page_mask = old_mask | leaf_mask;
        if old_mask == 0 {
            self.mark_upper_path(page_id);
        }
    }

    #[inline(always)]
    pub(super) fn commit_page_accesses(&mut self, accesses: &[PageAccess]) {
        for &access in accesses {
            self.mark_existing_page(access.page_id as usize, access.leaf_mask);
        }
    }

    #[inline(always)]
    fn mark_upper_path(&mut self, page_id: usize) {
        let mut node = (1usize << self.upper_height) + page_id;
        while node > 1 {
            node >>= 1;
            if !self.upper_nodes.insert_persistent(node) {
                break;
            }
        }
    }
}

#[inline(always)]
pub(super) fn leaf_mask_range(start: u32, end: u32) -> u64 {
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

    // At each level, collapse child occupancy into one representative bit per
    // parent group, then count groups reached by added leaves that were empty
    // in the old segment-local mask.
    //
    // ```text
    // leaves:       0 1   1 0   0 0   1 1
    // parents:       1     1     0     1
    // new parents:   (added_parent_bits & !old_parent_bits).count_ones()
    // ```
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

    // For one new leaf, each aligned group that was empty creates exactly one
    // new ancestor at that group size.
    let mut nodes = 0;
    nodes += u32::from(aligned_group_is_empty::<2>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<4>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<8>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<16>(old_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<32>(old_mask, leaf));
    nodes
}

#[inline(always)]
fn add_leaf_level_delta_with_default<const GROUP_SIZE: u32, const LEVEL: usize>(
    old_mask: u64,
    committed_mask: u64,
    leaf: u32,
    default_old: &mut DefaultOldCounts,
) -> u32 {
    if aligned_group_is_empty::<GROUP_SIZE>(old_mask, leaf) {
        if aligned_group_is_empty::<GROUP_SIZE>(committed_mask, leaf) {
            default_old.merkle_nodes += 1;
            default_old.merkle_node_levels |= 1u64 << LEVEL;
        }
        1
    } else {
        0
    }
}

#[inline(always)]
fn local_merkle_nodes_added_leaf_with_default(
    old_mask: u64,
    leaf: u32,
    committed_mask: u64,
) -> (u32, DefaultOldCounts) {
    debug_assert!(leaf < u64::BITS);

    if old_mask == 0 && committed_mask == 0 {
        return (
            MEMORY_PAGE_BITS as u32,
            DefaultOldCounts {
                leaves: 0,
                merkle_nodes: MEMORY_PAGE_BITS as u32,
                merkle_node_levels: (1u64 << (MEMORY_PAGE_BITS + 1)) - 2,
            },
        );
    }

    let mut default_old = DefaultOldCounts::default();
    let mut nodes = 0;
    nodes +=
        add_leaf_level_delta_with_default::<2, 1>(old_mask, committed_mask, leaf, &mut default_old);
    nodes +=
        add_leaf_level_delta_with_default::<4, 2>(old_mask, committed_mask, leaf, &mut default_old);
    nodes +=
        add_leaf_level_delta_with_default::<8, 3>(old_mask, committed_mask, leaf, &mut default_old);
    nodes += add_leaf_level_delta_with_default::<16, 4>(
        old_mask,
        committed_mask,
        leaf,
        &mut default_old,
    );
    nodes += add_leaf_level_delta_with_default::<32, 5>(
        old_mask,
        committed_mask,
        leaf,
        &mut default_old,
    );

    if old_mask == 0 {
        nodes += 1;
        if committed_mask == 0 {
            default_old.merkle_nodes += 1;
            default_old.merkle_node_levels |= 1u64 << MEMORY_PAGE_BITS;
        }
    }

    (nodes, default_old)
}

#[inline(always)]
fn add_level_delta_with_default<const SHIFT: u32, const MASK: u64, const LEVEL: usize>(
    old_mask: &mut u64,
    added_mask: &mut u64,
    committed_mask: &mut u64,
    default_old: &mut DefaultOldCounts,
) -> u32 {
    *old_mask = (*old_mask | (*old_mask >> SHIFT)) & MASK;
    *added_mask = (*added_mask | (*added_mask >> SHIFT)) & MASK;
    *committed_mask = (*committed_mask | (*committed_mask >> SHIFT)) & MASK;
    let new_nodes = *added_mask & !*old_mask;
    let default_nodes = new_nodes & !*committed_mask;
    default_old.merkle_nodes += default_nodes.count_ones();
    default_old.merkle_node_levels |= u64::from(default_nodes != 0) << LEVEL;
    new_nodes.count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta_with_default(
    mut old_mask: u64,
    mut added_mask: u64,
    mut committed_mask: u64,
) -> (u32, DefaultOldCounts) {
    debug_assert_ne!(added_mask, 0);
    debug_assert_eq!(old_mask & added_mask, 0);

    let mut default_old = DefaultOldCounts::default();
    let mut nodes = 0;
    nodes += add_level_delta_with_default::<1, FIRST_BIT_PER_PAIR, 1>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<2, FIRST_BIT_PER_NIBBLE, 2>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<4, FIRST_BIT_PER_BYTE, 3>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<8, FIRST_BIT_PER_U16, 4>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<16, FIRST_BIT_PER_U32, 5>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<32, FIRST_BIT_PER_U64, 6>(
        &mut old_mask,
        &mut added_mask,
        &mut committed_mask,
        &mut default_old,
    );

    (nodes, default_old)
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
    fn test_local_merkle_nodes_doc_example() {
        let mut tracker = MemoryPageTracker::new(0);
        let occupancy = MemoryOccupancyTracker::new(0);
        let mut nodes = 0;
        nodes += tracker.insert(0, 1 << 0, &occupancy).merkle_nodes;
        assert_eq!(nodes, 6);
        nodes += tracker.insert(0, 1 << 4, &occupancy).merkle_nodes;
        assert_eq!(nodes, 8);
        nodes += tracker.insert(0, 1 << 2, &occupancy).merkle_nodes;
        assert_eq!(nodes, 9);
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
        let occupancy = MemoryOccupancyTracker::new(3);
        assert_ne!(tracker.insert(0, 1 << 0, &occupancy).added_mask, 0);
        assert_eq!(
            tracker.insert(0, 1 << 0, &occupancy),
            MemoryHeightDelta::default()
        );
    }

    #[test]
    fn test_memory_page_tracker_clear_resets_touched_state() {
        let mut tracker = MemoryPageTracker::new(3);
        let occupancy = MemoryOccupancyTracker::new(3);
        let first = tracker.insert(0, 1 << 0, &occupancy);
        let second = tracker.insert(7, 1 << 63, &occupancy);
        assert!(first.leaves + second.leaves > 0);
        assert!(first.merkle_nodes + second.merkle_nodes > 0);

        tracker.clear();

        let after_clear = tracker.insert(0, 1 << 0, &occupancy);
        assert_eq!(after_clear.leaves, 1);
        assert!(after_clear.merkle_nodes > 0);
    }

    #[test]
    fn test_adjacent_pages_share_upper_ancestors() {
        let mut tracker = MemoryPageTracker::new(3);
        let occupancy = MemoryOccupancyTracker::new(3);
        tracker.insert(0, 1, &occupancy);
        let second = tracker.insert(1, 1, &occupancy);
        assert_eq!(second.merkle_nodes, MEMORY_PAGE_BITS as u32);
    }

    #[test]
    fn test_checkpoint_commit_keeps_occupied_counts() {
        let mut occupancy = MemoryOccupancyTracker::new(1);
        let mut tracker = MemoryPageTracker::new(1);

        let first = tracker.insert(0, 1, &occupancy).default_old;
        occupancy.commit_page_accesses(&[PageAccess {
            page_id: 0,
            leaf_mask: 1,
        }]);
        tracker.clear();
        let second = tracker.insert(0, 1, &occupancy).default_old;

        assert!(first.leaves > 0);
        assert_eq!(second, DefaultOldCounts::default());
    }
}
