use openvm_instructions::MEMORY_PAGE_BITS;

// Masks with the first bit set in each aligned group of leaves. These represent
// group occupancy at each page-local Merkle level.
const FIRST_LEAF_PER_2_LEAVES: u64 = 0x5555_5555_5555_5555;
const FIRST_LEAF_PER_4_LEAVES: u64 = 0x1111_1111_1111_1111;
const FIRST_LEAF_PER_8_LEAVES: u64 = 0x0101_0101_0101_0101;
const FIRST_LEAF_PER_16_LEAVES: u64 = 0x0001_0001_0001_0001;
const FIRST_LEAF_PER_32_LEAVES: u64 = 0x0000_0001_0000_0001;
const FIRST_LEAF_PER_64_LEAVES: u64 = 0x0000_0000_0000_0001;

/// Old-side leaves and Merkle nodes that were still canonical default values
/// when the current segment started.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DefaultOldAccounting {
    pub(super) default_leaves: u32,
    pub(super) default_merkle_nodes: u32,
    merkle_level_mask: u64,
}

impl DefaultOldAccounting {
    #[inline(always)]
    pub(super) fn add(&mut self, other: Self) {
        self.default_leaves += other.default_leaves;
        self.default_merkle_nodes += other.default_merkle_nodes;
        self.merkle_level_mask |= other.merkle_level_mask;
    }

    #[inline(always)]
    pub(super) fn estimated_poseidon_rows(self) -> u32 {
        u32::from(self.default_leaves != 0) + self.merkle_level_mask.count_ones()
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
    fn insert_clearable(&mut self, index: usize) -> bool {
        // Segment-local sets use dirty-word tracking so `clear` only touches
        // words written in the segment.
        self.insert_impl::<true>(index)
    }

    #[inline(always)]
    fn insert_committed(&mut self, index: usize) -> bool {
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
pub(super) struct MemoryAccountingDelta {
    pub(super) new_leaves: u32,
    pub(super) new_merkle_nodes: u32,
    pub(super) newly_charged_leaf_mask: u64,
    pub(super) default_old: DefaultOldAccounting,
}

/// Segment-local page accounting.
///
/// This tracker is cleared whenever a new segment starts. It answers: "which
/// leaves and Merkle nodes have already been charged in this segment?"
/// `CommittedMemoryOccupancyTracker` is passed into `insert` only to classify
/// old-side nodes as default or non-default.
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
pub(super) struct SegmentMemoryPageTracker {
    segment_leaf_masks: Box<[u64]>,
    dirty_page_ids: Vec<usize>,
    segment_upper_nodes: BitSet,
    upper_height: usize,
}

impl SegmentMemoryPageTracker {
    pub(super) fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            segment_leaf_masks: vec![0; num_pages].into_boxed_slice(),
            dirty_page_ids: Vec::new(),
            segment_upper_nodes: BitSet::new(num_pages),
            upper_height,
        }
    }

    #[inline(always)]
    pub(super) fn clear(&mut self) {
        for page_id in self.dirty_page_ids.drain(..) {
            // SAFETY: dirty_page_ids entries are page indices previously written by this tracker.
            unsafe {
                *self.segment_leaf_masks.get_unchecked_mut(page_id) = 0;
            }
        }
        self.segment_upper_nodes.clear();
    }

    #[inline(always)]
    pub(super) fn insert(
        &mut self,
        page_id: usize,
        leaf_mask: u64,
        occupancy_tracker: &CommittedMemoryOccupancyTracker,
    ) -> MemoryAccountingDelta {
        debug_assert!(page_id < self.segment_leaf_masks.len());
        debug_assert!(leaf_mask != 0);

        let segment_leaf_mask = unsafe { self.segment_leaf_masks.get_unchecked_mut(page_id) };
        let segment_leaf_mask_before = *segment_leaf_mask;
        let segment_leaf_mask_after = segment_leaf_mask_before | leaf_mask;
        if segment_leaf_mask_after == segment_leaf_mask_before {
            return MemoryAccountingDelta::default();
        }

        *segment_leaf_mask = segment_leaf_mask_after;
        if segment_leaf_mask_before == 0 {
            self.dirty_page_ids.push(page_id);
        }

        let newly_charged_leaf_mask = leaf_mask & !segment_leaf_mask_before;
        let committed_leaf_mask = occupancy_tracker.page_mask(page_id);
        let default_old_leaf_mask = newly_charged_leaf_mask & !committed_leaf_mask;
        let single_added_leaf = newly_charged_leaf_mask.is_power_of_two();
        let new_leaves = if single_added_leaf {
            1
        } else {
            newly_charged_leaf_mask.count_ones()
        };

        let (mut new_merkle_nodes, mut default_old) = if default_old_leaf_mask == 0 {
            (
                if single_added_leaf {
                    local_merkle_nodes_added_leaf(
                        segment_leaf_mask_before,
                        newly_charged_leaf_mask.trailing_zeros(),
                    )
                } else {
                    local_merkle_nodes_delta(segment_leaf_mask_before, newly_charged_leaf_mask)
                },
                DefaultOldAccounting::default(),
            )
        } else if single_added_leaf {
            local_merkle_nodes_added_leaf_with_default(
                segment_leaf_mask_before,
                newly_charged_leaf_mask.trailing_zeros(),
                committed_leaf_mask,
            )
        } else {
            local_merkle_nodes_delta_with_default(
                segment_leaf_mask_before,
                newly_charged_leaf_mask,
                committed_leaf_mask,
            )
        };
        if default_old_leaf_mask != 0 {
            default_old.default_leaves = if single_added_leaf {
                1
            } else {
                default_old_leaf_mask.count_ones()
            };
        }

        if segment_leaf_mask_before == 0 {
            if committed_leaf_mask == 0 {
                let (upper_nodes, upper_default_old) =
                    self.insert_upper_path_with_default_old(page_id, occupancy_tracker);
                new_merkle_nodes += upper_nodes;
                default_old.add(upper_default_old);
            } else {
                new_merkle_nodes += self.insert_upper_path_without_default_old(page_id);
            }
        }
        MemoryAccountingDelta {
            new_leaves,
            new_merkle_nodes,
            newly_charged_leaf_mask,
            default_old,
        }
    }

    #[inline(always)]
    fn insert_upper_path_with_default_old(
        &mut self,
        page_id: usize,
        occupancy_tracker: &CommittedMemoryOccupancyTracker,
    ) -> (u32, DefaultOldAccounting) {
        let mut count = 0;
        let mut default_old = DefaultOldAccounting::default();
        let mut node = (1usize << self.upper_height) + page_id;
        let mut height = MEMORY_PAGE_BITS + 1;
        while node > 1 {
            node >>= 1;
            if self.segment_upper_nodes.insert_clearable(node) {
                count += 1;
                if !occupancy_tracker.upper_contains(node) {
                    default_old.default_merkle_nodes += 1;
                    default_old.merkle_level_mask |= 1u64 << height;
                }
            } else {
                break;
            }
            height += 1;
        }
        (count, default_old)
    }

    #[inline(always)]
    fn insert_upper_path_without_default_old(&mut self, page_id: usize) -> u32 {
        let mut count = 0;
        let mut node = (1usize << self.upper_height) + page_id;
        while node > 1 {
            node >>= 1;
            if self.segment_upper_nodes.insert_clearable(node) {
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
pub(super) struct CommittedMemoryOccupancyTracker {
    committed_leaf_masks: Box<[u64]>,
    committed_upper_nodes: BitSet,
    upper_height: usize,
}

impl CommittedMemoryOccupancyTracker {
    pub(super) fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            committed_leaf_masks: vec![0; num_pages].into_boxed_slice(),
            committed_upper_nodes: BitSet::new(num_pages),
            upper_height,
        }
    }

    #[inline(always)]
    pub(super) fn page_mask(&self, page_id: usize) -> u64 {
        debug_assert!(page_id < self.committed_leaf_masks.len());
        unsafe { *self.committed_leaf_masks.get_unchecked(page_id) }
    }

    #[inline(always)]
    fn upper_contains(&self, node: usize) -> bool {
        self.committed_upper_nodes.contains(node)
    }

    #[inline(always)]
    pub(super) fn mark_existing_page(&mut self, page_id: usize, leaf_mask: u64) {
        debug_assert!(page_id < self.committed_leaf_masks.len());
        debug_assert!(leaf_mask != 0);

        let committed_leaf_mask = unsafe { self.committed_leaf_masks.get_unchecked_mut(page_id) };
        let committed_leaf_mask_before = *committed_leaf_mask;
        *committed_leaf_mask = committed_leaf_mask_before | leaf_mask;
        if committed_leaf_mask_before == 0 {
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
            if !self.committed_upper_nodes.insert_committed(node) {
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
    segment_mask: &mut u64,
    newly_charged_mask: &mut u64,
) -> u32 {
    *segment_mask = (*segment_mask | (*segment_mask >> SHIFT)) & MASK;
    *newly_charged_mask = (*newly_charged_mask | (*newly_charged_mask >> SHIFT)) & MASK;
    (*newly_charged_mask & !*segment_mask).count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta(mut segment_leaf_mask: u64, mut newly_charged_leaf_mask: u64) -> u32 {
    debug_assert_ne!(newly_charged_leaf_mask, 0);
    debug_assert_eq!(segment_leaf_mask & newly_charged_leaf_mask, 0);

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
    nodes += add_level_delta::<1, FIRST_LEAF_PER_2_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );
    nodes += add_level_delta::<2, FIRST_LEAF_PER_4_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );
    nodes += add_level_delta::<4, FIRST_LEAF_PER_8_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );
    nodes += add_level_delta::<8, FIRST_LEAF_PER_16_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );
    nodes += add_level_delta::<16, FIRST_LEAF_PER_32_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );
    nodes += add_level_delta::<32, FIRST_LEAF_PER_64_LEAVES>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
    );

    nodes
}

#[inline(always)]
fn aligned_group_is_empty<const GROUP_SIZE: u32>(leaf_mask: u64, leaf: u32) -> bool {
    let group_start = leaf & !(GROUP_SIZE - 1);
    let group_mask = ((1u64 << GROUP_SIZE) - 1) << group_start;
    leaf_mask & group_mask == 0
}

#[inline(always)]
fn local_merkle_nodes_added_leaf(segment_leaf_mask: u64, leaf: u32) -> u32 {
    debug_assert!(leaf < u64::BITS);

    if segment_leaf_mask == 0 {
        return MEMORY_PAGE_BITS as u32;
    }

    // For one new leaf, each aligned group that was empty creates exactly one
    // new ancestor at that group size.
    let mut nodes = 0;
    nodes += u32::from(aligned_group_is_empty::<2>(segment_leaf_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<4>(segment_leaf_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<8>(segment_leaf_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<16>(segment_leaf_mask, leaf));
    nodes += u32::from(aligned_group_is_empty::<32>(segment_leaf_mask, leaf));
    nodes
}

#[inline(always)]
fn add_leaf_level_delta_with_default<const GROUP_SIZE: u32, const LEVEL: usize>(
    segment_leaf_mask: u64,
    committed_leaf_mask: u64,
    leaf: u32,
    default_old: &mut DefaultOldAccounting,
) -> u32 {
    if aligned_group_is_empty::<GROUP_SIZE>(segment_leaf_mask, leaf) {
        if aligned_group_is_empty::<GROUP_SIZE>(committed_leaf_mask, leaf) {
            default_old.default_merkle_nodes += 1;
            default_old.merkle_level_mask |= 1u64 << LEVEL;
        }
        1
    } else {
        0
    }
}

#[inline(always)]
fn local_merkle_nodes_added_leaf_with_default(
    segment_leaf_mask: u64,
    leaf: u32,
    committed_leaf_mask: u64,
) -> (u32, DefaultOldAccounting) {
    debug_assert!(leaf < u64::BITS);

    if segment_leaf_mask == 0 && committed_leaf_mask == 0 {
        return (
            MEMORY_PAGE_BITS as u32,
            DefaultOldAccounting {
                default_leaves: 0,
                default_merkle_nodes: MEMORY_PAGE_BITS as u32,
                merkle_level_mask: (1u64 << (MEMORY_PAGE_BITS + 1)) - 2,
            },
        );
    }

    let mut default_old = DefaultOldAccounting::default();
    let mut nodes = 0;
    nodes += add_leaf_level_delta_with_default::<2, 1>(
        segment_leaf_mask,
        committed_leaf_mask,
        leaf,
        &mut default_old,
    );
    nodes += add_leaf_level_delta_with_default::<4, 2>(
        segment_leaf_mask,
        committed_leaf_mask,
        leaf,
        &mut default_old,
    );
    nodes += add_leaf_level_delta_with_default::<8, 3>(
        segment_leaf_mask,
        committed_leaf_mask,
        leaf,
        &mut default_old,
    );
    nodes += add_leaf_level_delta_with_default::<16, 4>(
        segment_leaf_mask,
        committed_leaf_mask,
        leaf,
        &mut default_old,
    );
    nodes += add_leaf_level_delta_with_default::<32, 5>(
        segment_leaf_mask,
        committed_leaf_mask,
        leaf,
        &mut default_old,
    );

    if segment_leaf_mask == 0 {
        nodes += 1;
        if committed_leaf_mask == 0 {
            default_old.default_merkle_nodes += 1;
            default_old.merkle_level_mask |= 1u64 << MEMORY_PAGE_BITS;
        }
    }

    (nodes, default_old)
}

#[inline(always)]
fn add_level_delta_with_default<const SHIFT: u32, const MASK: u64, const LEVEL: usize>(
    segment_mask: &mut u64,
    newly_charged_mask: &mut u64,
    committed_level_mask: &mut u64,
    default_old: &mut DefaultOldAccounting,
) -> u32 {
    *segment_mask = (*segment_mask | (*segment_mask >> SHIFT)) & MASK;
    *newly_charged_mask = (*newly_charged_mask | (*newly_charged_mask >> SHIFT)) & MASK;
    *committed_level_mask = (*committed_level_mask | (*committed_level_mask >> SHIFT)) & MASK;
    let new_nodes = *newly_charged_mask & !*segment_mask;
    let default_nodes = new_nodes & !*committed_level_mask;
    default_old.default_merkle_nodes += default_nodes.count_ones();
    default_old.merkle_level_mask |= u64::from(default_nodes != 0) << LEVEL;
    new_nodes.count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta_with_default(
    mut segment_leaf_mask: u64,
    mut newly_charged_leaf_mask: u64,
    mut committed_leaf_mask: u64,
) -> (u32, DefaultOldAccounting) {
    debug_assert_ne!(newly_charged_leaf_mask, 0);
    debug_assert_eq!(segment_leaf_mask & newly_charged_leaf_mask, 0);

    let mut default_old = DefaultOldAccounting::default();
    let mut nodes = 0;
    nodes += add_level_delta_with_default::<1, FIRST_LEAF_PER_2_LEAVES, 1>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<2, FIRST_LEAF_PER_4_LEAVES, 2>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<4, FIRST_LEAF_PER_8_LEAVES, 3>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<8, FIRST_LEAF_PER_16_LEAVES, 4>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<16, FIRST_LEAF_PER_32_LEAVES, 5>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
        &mut default_old,
    );
    nodes += add_level_delta_with_default::<32, FIRST_LEAF_PER_64_LEAVES, 6>(
        &mut segment_leaf_mask,
        &mut newly_charged_leaf_mask,
        &mut committed_leaf_mask,
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
        let mut tracker = SegmentMemoryPageTracker::new(0);
        let committed_occupancy = CommittedMemoryOccupancyTracker::new(0);
        let mut nodes = 0;
        nodes += tracker
            .insert(0, 1 << 0, &committed_occupancy)
            .new_merkle_nodes;
        assert_eq!(nodes, 6);
        nodes += tracker
            .insert(0, 1 << 4, &committed_occupancy)
            .new_merkle_nodes;
        assert_eq!(nodes, 8);
        nodes += tracker
            .insert(0, 1 << 2, &committed_occupancy)
            .new_merkle_nodes;
        assert_eq!(nodes, 9);
    }

    #[test]
    fn test_local_merkle_nodes_added_matches_reference_delta() {
        let mut segment_leaf_mask = 0u64;
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
            let segment_leaf_mask_after = segment_leaf_mask | leaf_mask;
            if segment_leaf_mask_after != segment_leaf_mask {
                let newly_charged_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                let expected = reference_local_merkle_nodes(segment_leaf_mask_after)
                    - reference_local_merkle_nodes(segment_leaf_mask);
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, newly_charged_leaf_mask),
                    expected
                );
                segment_leaf_mask = segment_leaf_mask_after;
            }
        }

        segment_leaf_mask = 1;
        for leaf_mask in additions {
            let segment_leaf_mask_after = segment_leaf_mask | leaf_mask;
            if segment_leaf_mask_after != segment_leaf_mask {
                let newly_charged_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                let expected = reference_local_merkle_nodes(segment_leaf_mask_after)
                    - reference_local_merkle_nodes(segment_leaf_mask);
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, newly_charged_leaf_mask),
                    expected
                );
                segment_leaf_mask = segment_leaf_mask_after;
            }
        }

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1024 {
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            let segment_leaf_mask = seed;
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            let segment_leaf_mask_after = segment_leaf_mask | seed;
            if segment_leaf_mask_after != segment_leaf_mask {
                let newly_charged_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, newly_charged_leaf_mask),
                    reference_local_merkle_nodes(segment_leaf_mask_after)
                        - reference_local_merkle_nodes(segment_leaf_mask)
                );
            }
        }
    }

    #[test]
    fn test_page_mask_duplicate_leaf_does_not_change_counts() {
        let mut tracker = SegmentMemoryPageTracker::new(3);
        let committed_occupancy = CommittedMemoryOccupancyTracker::new(3);
        assert_ne!(
            tracker
                .insert(0, 1 << 0, &committed_occupancy)
                .newly_charged_leaf_mask,
            0
        );
        assert_eq!(
            tracker.insert(0, 1 << 0, &committed_occupancy),
            MemoryAccountingDelta::default()
        );
    }

    #[test]
    fn test_memory_page_tracker_clear_resets_touched_state() {
        let mut tracker = SegmentMemoryPageTracker::new(3);
        let committed_occupancy = CommittedMemoryOccupancyTracker::new(3);
        let first = tracker.insert(0, 1 << 0, &committed_occupancy);
        let second = tracker.insert(7, 1 << 63, &committed_occupancy);
        assert!(first.new_leaves + second.new_leaves > 0);
        assert!(first.new_merkle_nodes + second.new_merkle_nodes > 0);

        tracker.clear();

        let after_clear = tracker.insert(0, 1 << 0, &committed_occupancy);
        assert_eq!(after_clear.new_leaves, 1);
        assert!(after_clear.new_merkle_nodes > 0);
    }

    #[test]
    fn test_adjacent_pages_share_upper_ancestors() {
        let mut tracker = SegmentMemoryPageTracker::new(3);
        let committed_occupancy = CommittedMemoryOccupancyTracker::new(3);
        tracker.insert(0, 1, &committed_occupancy);
        let second = tracker.insert(1, 1, &committed_occupancy);
        assert_eq!(second.new_merkle_nodes, MEMORY_PAGE_BITS as u32);
    }

    #[test]
    fn test_checkpoint_commit_keeps_occupied_counts() {
        let mut committed_occupancy = CommittedMemoryOccupancyTracker::new(1);
        let mut tracker = SegmentMemoryPageTracker::new(1);

        let first = tracker.insert(0, 1, &committed_occupancy).default_old;
        committed_occupancy.commit_page_accesses(&[PageAccess {
            page_id: 0,
            leaf_mask: 1,
        }]);
        tracker.clear();
        let second = tracker.insert(0, 1, &committed_occupancy).default_old;

        assert!(first.default_leaves > 0);
        assert_eq!(second, DefaultOldAccounting::default());
    }
}
