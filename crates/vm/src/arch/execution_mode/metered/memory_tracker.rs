use openvm_instructions::metering::{PAGE_MASK_LEAF_BITS, PAGE_MASK_LEAF_BITS_U32};

// Masks with the first bit set in each aligned group of leaves. These represent
// group occupancy at each page-local Merkle level.
const FIRST_LEAF_PER_2_LEAVES: u64 = 0x5555_5555_5555_5555;
const FIRST_LEAF_PER_4_LEAVES: u64 = 0x1111_1111_1111_1111;
const FIRST_LEAF_PER_8_LEAVES: u64 = 0x0101_0101_0101_0101;
const FIRST_LEAF_PER_16_LEAVES: u64 = 0x0001_0001_0001_0001;
const FIRST_LEAF_PER_32_LEAVES: u64 = 0x0000_0001_0000_0001;
const FIRST_LEAF_PER_64_LEAVES: u64 = 0x0000_0000_0000_0001;

/// Leaves touched within one 64-leaf page.
///
/// `page_id` selects the page and bit `i` in `leaf_mask` selects leaf `i` in that page.
///
/// ```text
/// leaf in page:  0  1  2  3  ... 63
/// mask bit:      1  0  1  0  ...  0
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PageTouch {
    /// Index of the 64-leaf page in the memory tree.
    pub page_id: u32,
    /// Aligns `leaf_mask` and makes the shared 16-byte Rust/C layout explicit.
    pub padding: u32,
    /// Leaves touched in this page, with one bit per leaf.
    pub leaf_mask: u64,
}

/// Leaves and tree nodes that must be created because they were absent at the last checkpoint.
///
/// An absent leaf starts as zero. An absent internal node starts as the hash of a zero-filled
/// subtree. Equal starting hashes share a Poseidon2 row, so we need both the total number of new
/// nodes and the highest tree level at which one appeared.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FirstTouchCounts {
    /// Newly used leaves that were absent at the last checkpoint.
    pub(super) leaves: u32,
    /// Newly needed internal nodes that were absent at the last checkpoint.
    pub(super) merkle_nodes: u32,
    /// Highest level containing a new node; every lower level also contains one.
    max_merkle_height: u32,
}

/// Remembers which zero-filled-tree hashes have already been added to this segment's trace.
///
/// ```text
///                 hash₂
///                /     \
///             hash₁   hash₁
///             /  \     /  \
///            0    0   0    0
/// ```
///
/// All zero leaves share one row. All `hash₁` nodes share another row, and all `hash₂` nodes share
/// another. Therefore the tracker only needs to remember whether the leaf row was added and the
/// highest internal-node level added so far. A checkpoint does not reset it because the trace still
/// belongs to the same segment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DefaultPoseidonRowTracker {
    /// Whether the row for a zero leaf has already been added.
    seen_leaf: bool,
    /// Highest zero-filled internal-node level already added; lower levels are implied.
    max_merkle_height: u32,
}

/// Additional tree work caused by one page touch.
///
/// The first two counts determine the new Boundary and Merkle rows. `first_touches` is the part
/// that starts from zero rather than from memory already present at the last checkpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SegmentMemoryDelta {
    /// Leaves not previously touched in this segment.
    pub(super) segment_leaves: u32,
    /// Internal nodes not previously needed in this segment.
    pub(super) segment_merkle_nodes: u32,
    /// Bits from the input touch that were new to this segment.
    pub(super) new_segment_leaf_mask: u64,
    /// New leaves and nodes that were also absent at the last checkpoint.
    pub(super) first_touches: FirstTouchCounts,
}

/// Fixed-size bit set that records which words are nonzero.
#[derive(Debug)]
struct BitSet {
    /// Packed occupancy bits.
    words: Box<[u64]>,
    /// Word indices written since the last `clear`.
    dirty_words: Vec<usize>,
}

/// Leaves and tree nodes already counted in the current segment.
///
/// This is cleared when a new segment starts. It prevents repeated reads or writes from adding the
/// same leaf or tree node to the trace more than once.
///
/// Memory leaves form one sparse Merkle tree. Each page contains the 64 leaves
/// represented by one `u64` leaf mask:
///
/// ```text
///        [root]              height h
///        /    \
///      ...    ...
///      /        \
///   [page]    [page]         h - PAGE_MASK_LEAF_BITS nodes above each page
///   / .. \
///  L  ..  L                  PAGE_MASK_LEAF_BITS levels inside a 64-leaf page
/// ```
///
/// The tracker counts each newly touched leaf once, each newly required
/// page-local internal node once, and each upper-tree ancestor once across all
/// pages in the segment.
#[derive(Debug)]
pub(super) struct SegmentMemoryTracker {
    /// Touched leaf masks by page for the current segment.
    segment_leaf_masks: Box<[u64]>,
    /// Pages written since the segment started.
    dirty_page_ids: Vec<usize>,
    /// Upper-tree ancestors already counted in the current segment.
    segment_upper_nodes: BitSet,
    /// Number of Merkle levels above the 64-leaf page layer.
    upper_height: usize,
}

/// Leaves and tree nodes present at the last safe checkpoint.
///
/// If execution must end the current segment at that checkpoint, the next segment starts from this
/// state. Unlike `SegmentMemoryTracker`, this survives the segment change. It is updated only after
/// a safe checkpoint and is initially populated from nonzero program memory.
#[derive(Debug)]
pub(super) struct BaselineMemoryTracker {
    /// Leaf masks present in the baseline, stored by page.
    baseline_leaf_masks: Box<[u64]>,
    /// Indices of pages containing at least one baseline leaf.
    nonempty_page_ids: Vec<usize>,
    /// Upper-tree ancestors present in the baseline.
    baseline_upper_nodes: BitSet,
    /// Number of Merkle levels above the 64-leaf page layer.
    upper_height: usize,
}

impl FirstTouchCounts {
    #[inline(always)]
    pub(super) fn add(&mut self, other: Self) {
        self.leaves += other.leaves;
        self.merkle_nodes += other.merkle_nodes;
        self.max_merkle_height = self.max_merkle_height.max(other.max_merkle_height);
    }
}

impl DefaultPoseidonRowTracker {
    #[inline(always)]
    pub(super) fn count_new(&mut self, first_touches: FirstTouchCounts) -> u32 {
        let new_leaf_row = u32::from(first_touches.leaves != 0 && !self.seen_leaf);
        self.seen_leaf |= first_touches.leaves != 0;

        let new_merkle_rows = first_touches
            .max_merkle_height
            .saturating_sub(self.max_merkle_height);
        self.max_merkle_height = self.max_merkle_height.max(first_touches.max_merkle_height);
        new_leaf_row + new_merkle_rows
    }
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
        self.insert_impl(index)
    }

    #[inline(always)]
    fn insert_baseline(&mut self, index: usize) -> bool {
        // Baseline memory is monotonic and persists across checkpoints.
        self.insert_impl(index)
    }

    #[inline(always)]
    fn insert_impl(&mut self, index: usize) -> bool {
        let word_index = index >> 6;
        let bit_index = index & 63;
        let mask = 1u64 << bit_index;

        debug_assert!(word_index < self.words.len(), "BitSet index out of bounds");

        // SAFETY: word_index is derived from a memory address that is bounds-checked
        //         during memory access. The bitset is sized to accommodate all valid
        //         memory addresses, so word_index is always within bounds.
        let word = unsafe { self.words.get_unchecked_mut(word_index) };
        let previous_word = *word;
        let was_set = (previous_word & mask) != 0;
        if was_set {
            return false;
        }
        if previous_word == 0 {
            self.dirty_words.push(word_index);
        }
        *word = previous_word | mask;
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
            // SAFETY: dirty_words contains indices previously written in this BitSet.
            unsafe {
                *self.words.get_unchecked_mut(word_index) = 0;
            }
        }
    }
}

impl BitSet {
    fn clone_populated(&self) -> Self {
        let mut cloned = Self {
            words: vec![0; self.words.len()].into_boxed_slice(),
            dirty_words: self.dirty_words.clone(),
        };
        for &word_index in &self.dirty_words {
            // SAFETY: dirty_words contains indices previously written in this BitSet.
            unsafe {
                *cloned.words.get_unchecked_mut(word_index) = *self.words.get_unchecked(word_index);
            }
        }
        cloned
    }
}

impl Clone for SegmentMemoryTracker {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.upper_height);
        cloned.dirty_page_ids = self.dirty_page_ids.clone();
        for &page_id in &self.dirty_page_ids {
            // SAFETY: dirty_page_ids contains indices previously written in this tracker.
            unsafe {
                *cloned.segment_leaf_masks.get_unchecked_mut(page_id) =
                    *self.segment_leaf_masks.get_unchecked(page_id);
            }
        }
        cloned.segment_upper_nodes = self.segment_upper_nodes.clone_populated();
        cloned
    }
}

impl Clone for BaselineMemoryTracker {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.upper_height);
        cloned.nonempty_page_ids = self.nonempty_page_ids.clone();
        for &page_id in &self.nonempty_page_ids {
            // SAFETY: nonempty_page_ids contains indices previously written in this tracker.
            unsafe {
                *cloned.baseline_leaf_masks.get_unchecked_mut(page_id) =
                    *self.baseline_leaf_masks.get_unchecked(page_id);
            }
        }
        cloned.baseline_upper_nodes = self.baseline_upper_nodes.clone_populated();
        cloned
    }
}

impl SegmentMemoryTracker {
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
        baseline_memory: &BaselineMemoryTracker,
    ) -> SegmentMemoryDelta {
        debug_assert!(page_id < self.segment_leaf_masks.len());
        debug_assert!(leaf_mask != 0);

        let segment_leaf_mask = unsafe { self.segment_leaf_masks.get_unchecked_mut(page_id) };
        let segment_leaf_mask_before = *segment_leaf_mask;
        let segment_leaf_mask_after = segment_leaf_mask_before | leaf_mask;
        if segment_leaf_mask_after == segment_leaf_mask_before {
            return SegmentMemoryDelta::default();
        }

        *segment_leaf_mask = segment_leaf_mask_after;
        if segment_leaf_mask_before == 0 {
            self.dirty_page_ids.push(page_id);
        }

        let new_segment_leaf_mask = leaf_mask & !segment_leaf_mask_before;
        let baseline_leaf_mask = baseline_memory.page_mask(page_id);
        let first_touch_leaf_mask = new_segment_leaf_mask & !baseline_leaf_mask;
        let single_added_leaf = new_segment_leaf_mask.is_power_of_two();
        let segment_leaves = if single_added_leaf {
            1
        } else {
            new_segment_leaf_mask.count_ones()
        };

        let (mut segment_merkle_nodes, mut first_touches) = if first_touch_leaf_mask == 0 {
            let nodes = if single_added_leaf {
                local_merkle_nodes_added_leaf(
                    segment_leaf_mask_before,
                    new_segment_leaf_mask.trailing_zeros(),
                )
            } else {
                local_merkle_nodes_delta(segment_leaf_mask_before, new_segment_leaf_mask)
            };
            (nodes, FirstTouchCounts::default())
        } else if single_added_leaf {
            local_merkle_nodes_added_leaf_with_first_touches(
                segment_leaf_mask_before,
                new_segment_leaf_mask.trailing_zeros(),
                baseline_leaf_mask,
            )
        } else {
            local_merkle_nodes_delta_with_first_touches(
                segment_leaf_mask_before,
                new_segment_leaf_mask,
                baseline_leaf_mask,
            )
        };
        if first_touch_leaf_mask != 0 {
            first_touches.leaves = if single_added_leaf {
                1
            } else {
                first_touch_leaf_mask.count_ones()
            };
        }

        if segment_leaf_mask_before == 0 {
            if baseline_leaf_mask == 0 {
                let (upper_nodes, upper_first_touches) =
                    self.insert_upper_path_with_first_touches(page_id, baseline_memory);
                segment_merkle_nodes += upper_nodes;
                first_touches.add(upper_first_touches);
            } else {
                segment_merkle_nodes += self.insert_upper_path(page_id);
            }
        }
        SegmentMemoryDelta {
            segment_leaves,
            segment_merkle_nodes,
            new_segment_leaf_mask,
            first_touches,
        }
    }

    #[inline(always)]
    fn insert_upper_path_with_first_touches(
        &mut self,
        page_id: usize,
        baseline_memory: &BaselineMemoryTracker,
    ) -> (u32, FirstTouchCounts) {
        let mut count = 0;
        let mut first_touches = FirstTouchCounts::default();
        let mut node = (1usize << self.upper_height) + page_id;
        let mut height = PAGE_MASK_LEAF_BITS + 1;
        while node > 1 {
            node >>= 1;
            if self.segment_upper_nodes.insert_clearable(node) {
                count += 1;
                if !baseline_memory.upper_contains(node) {
                    first_touches.merkle_nodes += 1;
                    first_touches.max_merkle_height = height as u32;
                }
            } else {
                break;
            }
            height += 1;
        }
        (count, first_touches)
    }

    #[inline(always)]
    fn insert_upper_path(&mut self, page_id: usize) -> u32 {
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

impl BaselineMemoryTracker {
    pub(super) fn new(upper_height: usize) -> Self {
        let num_pages = 1 << upper_height;
        Self {
            baseline_leaf_masks: vec![0; num_pages].into_boxed_slice(),
            nonempty_page_ids: Vec::new(),
            baseline_upper_nodes: BitSet::new(num_pages),
            upper_height,
        }
    }

    #[inline(always)]
    pub(super) fn page_mask(&self, page_id: usize) -> u64 {
        debug_assert!(page_id < self.baseline_leaf_masks.len());
        unsafe { *self.baseline_leaf_masks.get_unchecked(page_id) }
    }

    #[inline(always)]
    fn upper_contains(&self, node: usize) -> bool {
        self.baseline_upper_nodes.contains(node)
    }

    #[inline(always)]
    pub(super) fn mark_existing_page(&mut self, page_id: usize, leaf_mask: u64) {
        debug_assert!(page_id < self.baseline_leaf_masks.len());
        debug_assert!(leaf_mask != 0);

        let baseline_leaf_mask = unsafe { self.baseline_leaf_masks.get_unchecked_mut(page_id) };
        let baseline_leaf_mask_before = *baseline_leaf_mask;
        *baseline_leaf_mask = baseline_leaf_mask_before | leaf_mask;
        if baseline_leaf_mask_before == 0 {
            self.nonempty_page_ids.push(page_id);
            self.mark_upper_path(page_id);
        }
    }

    #[inline(always)]
    pub(super) fn add_page_touches(&mut self, touches: &[PageTouch]) {
        for &touch in touches {
            self.mark_existing_page(touch.page_id as usize, touch.leaf_mask);
        }
    }

    #[inline(always)]
    fn mark_upper_path(&mut self, page_id: usize) {
        let mut node = (1usize << self.upper_height) + page_id;
        while node > 1 {
            node >>= 1;
            if !self.baseline_upper_nodes.insert_baseline(node) {
                break;
            }
        }
    }
}

/// Returns a mask with bits in `[start, end)` set.
#[inline(always)]
pub(super) fn leaf_mask_range(start: u32, end: u32) -> u64 {
    debug_assert!(start < end);
    debug_assert!(end <= u64::BITS);
    (u64::MAX << start) & (u64::MAX >> (u64::BITS - end))
}

#[inline(always)]
fn add_level_delta<const SHIFT: u32, const MASK: u64>(
    segment_mask: &mut u64,
    new_segment_mask: &mut u64,
) -> u32 {
    *segment_mask = (*segment_mask | (*segment_mask >> SHIFT)) & MASK;
    *new_segment_mask = (*new_segment_mask | (*new_segment_mask >> SHIFT)) & MASK;
    (*new_segment_mask & !*segment_mask).count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta(mut segment_leaf_mask: u64, mut new_segment_leaf_mask: u64) -> u32 {
    debug_assert_ne!(new_segment_leaf_mask, 0);
    debug_assert_eq!(segment_leaf_mask & new_segment_leaf_mask, 0);

    // At each level, collapse child occupancy into one representative bit per
    // parent group, then count groups reached by new segment leaves that were
    // empty in the segment mask.
    //
    // ```text
    // leaves:       0 1   1 0   0 0   1 1
    // parents:       1     1     0     1
    // new parents:   (added_parent_bits & !segment_parent_bits).count_ones()
    // ```
    let mut nodes = 0;
    nodes += add_level_delta::<1, FIRST_LEAF_PER_2_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
    );
    nodes += add_level_delta::<2, FIRST_LEAF_PER_4_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
    );
    nodes += add_level_delta::<4, FIRST_LEAF_PER_8_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
    );
    nodes += add_level_delta::<8, FIRST_LEAF_PER_16_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
    );
    nodes += add_level_delta::<16, FIRST_LEAF_PER_32_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
    );
    nodes += add_level_delta::<32, FIRST_LEAF_PER_64_LEAVES>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
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
        return PAGE_MASK_LEAF_BITS_U32;
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
fn add_leaf_level_delta_with_first_touch<const GROUP_SIZE: u32, const LEVEL: u32>(
    segment_leaf_mask: u64,
    baseline_leaf_mask: u64,
    leaf: u32,
    first_touches: &mut FirstTouchCounts,
) -> u32 {
    if aligned_group_is_empty::<GROUP_SIZE>(segment_leaf_mask, leaf) {
        if aligned_group_is_empty::<GROUP_SIZE>(baseline_leaf_mask, leaf) {
            first_touches.merkle_nodes += 1;
            first_touches.max_merkle_height = LEVEL;
        }
        1
    } else {
        0
    }
}

#[inline(always)]
fn local_merkle_nodes_added_leaf_with_first_touches(
    segment_leaf_mask: u64,
    leaf: u32,
    baseline_leaf_mask: u64,
) -> (u32, FirstTouchCounts) {
    debug_assert!(leaf < u64::BITS);

    if segment_leaf_mask == 0 && baseline_leaf_mask == 0 {
        return (
            PAGE_MASK_LEAF_BITS_U32,
            FirstTouchCounts {
                leaves: 0,
                merkle_nodes: PAGE_MASK_LEAF_BITS_U32,
                max_merkle_height: PAGE_MASK_LEAF_BITS_U32,
            },
        );
    }

    let mut first_touches = FirstTouchCounts::default();
    let mut nodes = 0;
    nodes += add_leaf_level_delta_with_first_touch::<2, 1>(
        segment_leaf_mask,
        baseline_leaf_mask,
        leaf,
        &mut first_touches,
    );
    nodes += add_leaf_level_delta_with_first_touch::<4, 2>(
        segment_leaf_mask,
        baseline_leaf_mask,
        leaf,
        &mut first_touches,
    );
    nodes += add_leaf_level_delta_with_first_touch::<8, 3>(
        segment_leaf_mask,
        baseline_leaf_mask,
        leaf,
        &mut first_touches,
    );
    nodes += add_leaf_level_delta_with_first_touch::<16, 4>(
        segment_leaf_mask,
        baseline_leaf_mask,
        leaf,
        &mut first_touches,
    );
    nodes += add_leaf_level_delta_with_first_touch::<32, 5>(
        segment_leaf_mask,
        baseline_leaf_mask,
        leaf,
        &mut first_touches,
    );

    if segment_leaf_mask == 0 {
        nodes += 1;
        if baseline_leaf_mask == 0 {
            first_touches.merkle_nodes += 1;
            first_touches.max_merkle_height = PAGE_MASK_LEAF_BITS_U32;
        }
    }

    (nodes, first_touches)
}

#[inline(always)]
fn add_level_delta_with_first_touches<const SHIFT: u32, const MASK: u64, const LEVEL: u32>(
    segment_mask: &mut u64,
    new_segment_mask: &mut u64,
    baseline_level_mask: &mut u64,
    first_touches: &mut FirstTouchCounts,
) -> u32 {
    *segment_mask = (*segment_mask | (*segment_mask >> SHIFT)) & MASK;
    *new_segment_mask = (*new_segment_mask | (*new_segment_mask >> SHIFT)) & MASK;
    *baseline_level_mask = (*baseline_level_mask | (*baseline_level_mask >> SHIFT)) & MASK;
    let new_nodes = *new_segment_mask & !*segment_mask;
    let first_touch_nodes = new_nodes & !*baseline_level_mask;
    first_touches.merkle_nodes += first_touch_nodes.count_ones();
    if first_touch_nodes != 0 {
        first_touches.max_merkle_height = LEVEL;
    }
    new_nodes.count_ones()
}

#[inline(always)]
fn local_merkle_nodes_delta_with_first_touches(
    mut segment_leaf_mask: u64,
    mut new_segment_leaf_mask: u64,
    mut baseline_leaf_mask: u64,
) -> (u32, FirstTouchCounts) {
    debug_assert_ne!(new_segment_leaf_mask, 0);
    debug_assert_eq!(segment_leaf_mask & new_segment_leaf_mask, 0);

    let mut first_touches = FirstTouchCounts::default();
    let mut nodes = 0;
    nodes += add_level_delta_with_first_touches::<1, FIRST_LEAF_PER_2_LEAVES, 1>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );
    nodes += add_level_delta_with_first_touches::<2, FIRST_LEAF_PER_4_LEAVES, 2>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );
    nodes += add_level_delta_with_first_touches::<4, FIRST_LEAF_PER_8_LEAVES, 3>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );
    nodes += add_level_delta_with_first_touches::<8, FIRST_LEAF_PER_16_LEAVES, 4>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );
    nodes += add_level_delta_with_first_touches::<16, FIRST_LEAF_PER_32_LEAVES, 5>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );
    nodes += add_level_delta_with_first_touches::<32, FIRST_LEAF_PER_64_LEAVES, 6>(
        &mut segment_leaf_mask,
        &mut new_segment_leaf_mask,
        &mut baseline_leaf_mask,
        &mut first_touches,
    );

    (nodes, first_touches)
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
        for _ in 0..PAGE_MASK_LEAF_BITS {
            level_mask = reference_parent_mask(level_mask);
            nodes += level_mask.count_ones();
        }
        nodes
    }

    #[test]
    fn test_local_merkle_nodes_doc_example() {
        let mut tracker = SegmentMemoryTracker::new(0);
        let baseline_memory = BaselineMemoryTracker::new(0);
        let mut nodes = 0;
        nodes += tracker
            .insert(0, 1 << 0, &baseline_memory)
            .segment_merkle_nodes;
        assert_eq!(nodes, 6);
        nodes += tracker
            .insert(0, 1 << 4, &baseline_memory)
            .segment_merkle_nodes;
        assert_eq!(nodes, 8);
        nodes += tracker
            .insert(0, 1 << 2, &baseline_memory)
            .segment_merkle_nodes;
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
                let new_segment_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                let expected = reference_local_merkle_nodes(segment_leaf_mask_after)
                    - reference_local_merkle_nodes(segment_leaf_mask);
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, new_segment_leaf_mask),
                    expected
                );
                segment_leaf_mask = segment_leaf_mask_after;
            }
        }

        segment_leaf_mask = 1;
        for leaf_mask in additions {
            let segment_leaf_mask_after = segment_leaf_mask | leaf_mask;
            if segment_leaf_mask_after != segment_leaf_mask {
                let new_segment_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                let expected = reference_local_merkle_nodes(segment_leaf_mask_after)
                    - reference_local_merkle_nodes(segment_leaf_mask);
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, new_segment_leaf_mask),
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
                let new_segment_leaf_mask = segment_leaf_mask_after ^ segment_leaf_mask;
                assert_eq!(
                    local_merkle_nodes_delta(segment_leaf_mask, new_segment_leaf_mask),
                    reference_local_merkle_nodes(segment_leaf_mask_after)
                        - reference_local_merkle_nodes(segment_leaf_mask)
                );
            }
        }
    }

    #[test]
    fn test_page_mask_duplicate_leaf_does_not_change_counts() {
        let mut tracker = SegmentMemoryTracker::new(3);
        let baseline_memory = BaselineMemoryTracker::new(3);
        assert_ne!(
            tracker
                .insert(0, 1 << 0, &baseline_memory)
                .new_segment_leaf_mask,
            0
        );
        assert_eq!(
            tracker.insert(0, 1 << 0, &baseline_memory),
            SegmentMemoryDelta::default()
        );
    }

    #[test]
    fn test_segment_memory_clear_resets_touched_state() {
        let mut tracker = SegmentMemoryTracker::new(3);
        let baseline_memory = BaselineMemoryTracker::new(3);
        let first = tracker.insert(0, 1 << 0, &baseline_memory);
        let second = tracker.insert(7, 1 << 63, &baseline_memory);
        assert!(first.segment_leaves + second.segment_leaves > 0);
        assert!(first.segment_merkle_nodes + second.segment_merkle_nodes > 0);

        tracker.clear();

        let after_clear = tracker.insert(0, 1 << 0, &baseline_memory);
        assert_eq!(after_clear.segment_leaves, 1);
        assert!(after_clear.segment_merkle_nodes > 0);
    }

    #[test]
    fn test_adjacent_pages_share_upper_ancestors() {
        let mut tracker = SegmentMemoryTracker::new(3);
        let baseline_memory = BaselineMemoryTracker::new(3);
        tracker.insert(0, 1, &baseline_memory);
        let second = tracker.insert(1, 1, &baseline_memory);
        assert_eq!(second.segment_merkle_nodes, PAGE_MASK_LEAF_BITS_U32);
    }

    #[test]
    fn test_baseline_memory_suppresses_second_first_touch() {
        let mut baseline_memory = BaselineMemoryTracker::new(1);
        let mut tracker = SegmentMemoryTracker::new(1);

        let first = tracker.insert(0, 1, &baseline_memory).first_touches;
        baseline_memory.add_page_touches(&[PageTouch {
            page_id: 0,
            padding: 0,
            leaf_mask: 1,
        }]);
        tracker.clear();
        let second = tracker.insert(0, 1, &baseline_memory).first_touches;

        assert!(first.leaves > 0);
        assert_eq!(second, FirstTouchCounts::default());
    }

    #[test]
    fn test_segment_tracker_clone_preserves_populated_entries() {
        let baseline_memory = BaselineMemoryTracker::new(3);
        let mut tracker = SegmentMemoryTracker::new(3);
        tracker.insert(0, 1, &baseline_memory);
        tracker.insert(7, 1 << 63, &baseline_memory);

        let mut cloned = tracker.clone();
        assert_eq!(
            cloned.insert(0, 1, &baseline_memory),
            SegmentMemoryDelta::default()
        );
        assert_eq!(
            cloned.insert(7, 1 << 63, &baseline_memory),
            SegmentMemoryDelta::default()
        );
        assert_eq!(
            cloned.insert(1, 1, &baseline_memory),
            tracker.insert(1, 1, &baseline_memory)
        );
    }

    #[test]
    fn test_baseline_tracker_clone_preserves_populated_entries() {
        let mut baseline_memory = BaselineMemoryTracker::new(1);
        baseline_memory.mark_existing_page(0, 1);
        baseline_memory.mark_existing_page(0, 2);
        assert_eq!(baseline_memory.nonempty_page_ids, [0]);

        let cloned = baseline_memory.clone();
        let mut tracker = SegmentMemoryTracker::new(1);
        assert_eq!(
            tracker.insert(0, 0b11, &cloned).first_touches,
            FirstTouchCounts::default()
        );
        assert_ne!(
            tracker.insert(1, 1, &cloned).first_touches,
            FirstTouchCounts::default()
        );
    }

    #[test]
    fn test_default_poseidon_rows_only_charge_new_buckets() {
        let mut rows = DefaultPoseidonRowTracker::default();
        assert_eq!(
            rows.count_new(FirstTouchCounts {
                leaves: 1,
                merkle_nodes: 6,
                max_merkle_height: 6,
            }),
            7
        );
        assert_eq!(
            rows.count_new(FirstTouchCounts {
                leaves: 2,
                merkle_nodes: 3,
                max_merkle_height: 3,
            }),
            0
        );
        assert_eq!(
            rows.count_new(FirstTouchCounts {
                leaves: 0,
                merkle_nodes: 2,
                max_merkle_height: 8,
            }),
            2
        );
    }
}
