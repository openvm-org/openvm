use std::mem::size_of;

use openvm_instructions::{
    exe::SparseMemoryImage,
    metering::{PAGE_MASK_LEAF_BITS, SEGMENT_CHECK_INSNS},
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    DEFERRAL_AS, VM_DIGEST_WIDTH,
};
#[cfg(test)]
use openvm_instructions::{riscv::RV64_MEMORY_AS, PUBLIC_VALUES_AS};

pub use super::memory_tracker::PageTouch;
use super::memory_tracker::{
    leaf_mask_range, BaselineMemoryTracker, DefaultPoseidonRowTracker, FirstTouchCounts,
    SegmentMemoryTracker,
};
use crate::{
    arch::{SystemConfig, ADDR_SPACE_OFFSET, BOUNDARY_AIR_ID, MERKLE_AIR_ID, U16_CELL_SIZE},
    system::memory::dimensions::MemoryDimensions,
};

/// Upper bound on number of memory pages accessed per instruction. Used for buffer allocation.
const MAX_MEM_PAGE_OPS_PER_INSN: usize = 1 << 16;
// Initial allocation only. Correctness is preserved by `Vec::reserve` for large
// range accesses; this avoids preallocating the worst-case per-instruction page
// count for the common scalar-access path.
const INITIAL_CHECKPOINT_PAGE_ACCESSES_PER_INSN: usize = 16;
// Shift amounts from address-space pointer units to memory Merkle leaves.
const BYTE_PTRS_PER_LEAF_BITS: u32 = (U16_CELL_SIZE * VM_DIGEST_WIDTH).ilog2();
const DEFERRAL_PTRS_PER_LEAF_BITS: u32 = VM_DIGEST_WIDTH.ilog2();
const FIELD_ELEMENT_BYTES: u32 = size_of::<u32>() as u32;

/// Tracks which parts of memory contribute rows to the current segment.
///
/// It separately remembers what the current segment has used and what memory existed at the last
/// safe checkpoint. If the segment ends at that checkpoint, later touches are replayed into the
/// next segment. Repeated touches in one segment are counted only once.
#[derive(Clone, Debug)]
pub struct MemoryCtx {
    /// Memory tree dimensions used to map address-space ranges into global leaf ids.
    memory_dimensions: MemoryDimensions,
    /// Memory leaves and nodes already counted in the current segment.
    segment_memory: SegmentMemoryTracker,
    /// Memory leaves and nodes present in the baseline at the last checkpoint.
    baseline_memory: BaselineMemoryTracker,
    /// Touches since the last safe checkpoint, kept so they can be replayed into a new segment.
    pub page_indices_since_checkpoint: Vec<PageTouch>,
    /// Length mirror used by generated metered code that appends through the raw buffer.
    pub page_indices_since_checkpoint_len: usize,
    /// Number of buffered touches already included in the current row estimates.
    page_indices_applied_len: usize,
    /// Newly used leaves that will become part of the baseline at the next safe checkpoint.
    pending_baseline_updates: Vec<PageTouch>,
    /// Newly used leaves waiting to be added to the Boundary row estimate.
    pending_segment_leaves: u32,
    /// Newly needed tree nodes waiting to be added to the Merkle row estimate.
    pending_segment_merkle_nodes: u32,
    /// Pending leaves and nodes that were absent at the last checkpoint.
    pending_first_touches: FirstTouchCounts,
    /// Reusable default-hash rows already charged in the current segment.
    default_poseidon_rows: DefaultPoseidonRowTracker,
}

impl MemoryCtx {
    pub fn new(config: &SystemConfig) -> Self {
        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        let upper_height = merkle_height.saturating_sub(PAGE_MASK_LEAF_BITS);
        let checkpoint_capacity = Self::initial_checkpoint_capacity();

        Self {
            memory_dimensions,
            segment_memory: SegmentMemoryTracker::new(upper_height),
            baseline_memory: BaselineMemoryTracker::new(upper_height),
            page_indices_since_checkpoint: Vec::with_capacity(checkpoint_capacity),
            page_indices_since_checkpoint_len: 0,
            page_indices_applied_len: 0,
            pending_baseline_updates: Vec::with_capacity(checkpoint_capacity),
            pending_segment_leaves: 0,
            pending_segment_merkle_nodes: 0,
            pending_first_touches: FirstTouchCounts::default(),
            default_poseidon_rows: DefaultPoseidonRowTracker::default(),
        }
    }

    #[inline(always)]
    fn initial_checkpoint_capacity() -> usize {
        SEGMENT_CHECK_INSNS as usize * INITIAL_CHECKPOINT_PAGE_ACCESSES_PER_INSN
    }

    #[inline(always)]
    pub(crate) fn add_register_merkle_heights(&mut self) {
        self.update_boundary_merkle_heights(
            RV64_REGISTER_AS,
            0,
            (RV64_NUM_REGISTERS * RV64_REGISTER_NUM_LIMBS) as u32,
        );
    }

    /// Marks leaves containing nonzero program data as present before execution starts.
    pub(crate) fn seed_initial_memory(&mut self, initial_memory: &SparseMemoryImage) {
        for (&(addr_space, ptr), &byte) in initial_memory {
            if byte == 0 {
                continue;
            }
            // The sparse image is byte-addressed. One nonzero byte is enough
            // to make the containing Merkle leaf non-default.
            if addr_space == DEFERRAL_AS {
                self.mark_existing_memory_range(addr_space, ptr / FIELD_ELEMENT_BYTES, 1);
            } else {
                self.mark_existing_memory_range(addr_space, ptr, 1);
            }
        }
    }

    #[inline(always)]
    fn leaf_id_range(&self, address_space: u32, ptr: u32, size: u32) -> (u32, u32) {
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
        (start_leaf_id, end_leaf_id)
    }

    #[inline(always)]
    fn mark_existing_memory_range(&mut self, address_space: u32, ptr: u32, size: u32) {
        let (start_leaf_id, end_leaf_id) = self.leaf_id_range(address_space, ptr, size);
        let start_page_id = start_leaf_id >> PAGE_MASK_LEAF_BITS;
        let end_page_id = ((end_leaf_id - 1) >> PAGE_MASK_LEAF_BITS) + 1;

        for page_id in start_page_id..end_page_id {
            let page_start = page_id << PAGE_MASK_LEAF_BITS;
            let page_end = page_start + (1 << PAGE_MASK_LEAF_BITS);
            let start = start_leaf_id.max(page_start) - page_start;
            let end = end_leaf_id.min(page_end) - page_start;
            let leaf_mask = leaf_mask_range(start, end);
            self.baseline_memory
                .mark_existing_page(page_id as usize, leaf_mask);
        }
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
        let (start_leaf_id, end_leaf_id) = self.leaf_id_range(address_space, ptr, size);
        let num_leaves = end_leaf_id - start_leaf_id;
        let start_page_id = start_leaf_id >> PAGE_MASK_LEAF_BITS;

        if num_leaves == 1 {
            push_page_touch(
                &mut self.page_indices_since_checkpoint,
                start_page_id,
                1u64 << (start_leaf_id & ((1 << PAGE_MASK_LEAF_BITS) - 1)),
            );
            self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
            return;
        }

        let end_page_id = ((end_leaf_id - 1) >> PAGE_MASK_LEAF_BITS) + 1;
        let num_pages = (end_page_id - start_page_id) as usize;
        assert!(
            num_pages <= MAX_MEM_PAGE_OPS_PER_INSN,
            "more than {MAX_MEM_PAGE_OPS_PER_INSN} memory pages accessed in a single instruction"
        );
        if num_pages > 1 {
            self.page_indices_since_checkpoint.reserve(num_pages);
        }

        for page_id in start_page_id..end_page_id {
            let page_start = page_id << PAGE_MASK_LEAF_BITS;
            let page_end = page_start + (1 << PAGE_MASK_LEAF_BITS);
            let start = start_leaf_id.max(page_start) - page_start;
            let end = end_leaf_id.min(page_end) - page_start;
            let leaf_mask = leaf_mask_range(start, end);
            push_page_touch(&mut self.page_indices_since_checkpoint, page_id, leaf_mask);
        }
        self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
    }

    #[cfg(feature = "rvr")]
    #[inline(always)]
    pub(crate) fn apply_page_touches_with_offset(
        &mut self,
        page_offset: u32,
        touches: &[PageTouch],
    ) {
        let len = touches.len();
        let ptr = touches.as_ptr();
        for i in 0..len {
            // SAFETY: i is bounded by touches.len().
            let touch = unsafe { *ptr.add(i) };
            debug_assert!(touch.leaf_mask != 0);
            let page_id = page_offset + touch.page_id;
            let delta = self.segment_memory.insert(
                page_id as usize,
                touch.leaf_mask,
                &self.baseline_memory,
            );
            if delta.new_segment_leaf_mask != 0 {
                self.pending_segment_leaves += delta.segment_leaves;
                self.pending_segment_merkle_nodes += delta.segment_merkle_nodes;
                push_page_touch(
                    &mut self.pending_baseline_updates,
                    page_id,
                    delta.new_segment_leaf_mask,
                );
                self.pending_first_touches.add(delta.first_touches);
            }
        }
    }

    #[cfg(feature = "rvr")]
    #[inline(always)]
    pub(crate) fn reset_segment_without_replay(&mut self, trace_heights: &mut [u32]) {
        self.segment_memory.clear();
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_baseline_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_first_touches = FirstTouchCounts::default();
        self.default_poseidon_rows = DefaultPoseidonRowTracker::default();

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
        self.segment_memory.clear();
        self.page_indices_applied_len = 0;
        self.pending_baseline_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_first_touches = FirstTouchCounts::default();
        self.default_poseidon_rows = DefaultPoseidonRowTracker::default();

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

    /// Adds pending segment updates to the baseline.
    #[inline(always)]
    pub(crate) fn update_checkpoint(&mut self) {
        self.baseline_memory
            .add_page_touches(&self.pending_baseline_updates);
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_baseline_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_first_touches = FirstTouchCounts::default();
    }

    /// Applies memory height deltas recorded since the last checkpoint.
    ///
    /// - BOUNDARY_AIR: `2 * segment_leaves` rows
    /// - MERKLE_AIR:   `2 * segment_merkle_nodes` rows
    /// - Poseidon2:    hashes at the end of the segment plus hashes at its start
    ///
    /// A leaf or node absent at the checkpoint starts from the hash of zero-filled memory. Equal
    /// starting hashes share a row: one for all zero leaves and one for each internal-node height.
    /// Therefore:
    ///
    /// ```text
    /// Poseidon2 rows = end-of-segment hashes
    ///                + nonzero checkpoint hashes
    ///                + shared zero-filled hashes
    /// ```
    #[inline(always)]
    pub(crate) fn apply_height_updates(&mut self, trace_heights: &mut [u32]) {
        let mut leaves = self.pending_segment_leaves;
        let mut merkle_nodes = self.pending_segment_merkle_nodes;
        let mut first_touches = self.pending_first_touches;
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_first_touches = FirstTouchCounts::default();

        let len = self.page_indices_since_checkpoint.len();
        let ptr = self.page_indices_since_checkpoint.as_ptr();
        for i in self.page_indices_applied_len..len {
            // SAFETY: i is bounded by page_indices_since_checkpoint.len().
            let touch = unsafe { *ptr.add(i) };
            let delta = self.segment_memory.insert(
                touch.page_id as usize,
                touch.leaf_mask,
                &self.baseline_memory,
            );
            if delta.new_segment_leaf_mask != 0 {
                leaves += delta.segment_leaves;
                merkle_nodes += delta.segment_merkle_nodes;
                push_page_touch(
                    &mut self.pending_baseline_updates,
                    touch.page_id,
                    delta.new_segment_leaf_mask,
                );
                first_touches.add(delta.first_touches);
            }
        }
        self.page_indices_applied_len = len;

        if leaves == 0 && merkle_nodes == 0 {
            return;
        }

        debug_assert!(trace_heights.len() >= 2);
        let poseidon2_idx = trace_heights.len() - 2;
        let initial_default_poseidon_rows = self.default_poseidon_rows.count_new(first_touches);
        let initial_nondefault_poseidon_rows = (leaves + merkle_nodes)
            .saturating_sub(first_touches.leaves + first_touches.merkle_nodes);
        let poseidon2_rows = leaves
            + merkle_nodes
            + initial_nondefault_poseidon_rows
            + initial_default_poseidon_rows;
        // SAFETY: BOUNDARY_AIR_ID, MERKLE_AIR_ID, and poseidon2_idx are all within bounds
        unsafe {
            *trace_heights.get_unchecked_mut(BOUNDARY_AIR_ID) += leaves * 2;
            *trace_heights.get_unchecked_mut(poseidon2_idx) += poseidon2_rows;
            // Merkle AIR: 2 rows per internal node (init + final tree)
            *trace_heights.get_unchecked_mut(MERKLE_AIR_ID) += merkle_nodes * 2;
        }
    }
}

#[inline(always)]
fn push_page_touch(touches: &mut Vec<PageTouch>, page_id: u32, leaf_mask: u64) {
    debug_assert!(leaf_mask != 0);
    let len = touches.len();
    if len != 0 {
        // SAFETY: len is non-zero, so len - 1 is in bounds.
        let prev = unsafe { touches.get_unchecked_mut(len - 1) };
        if prev.page_id == page_id {
            prev.leaf_mask |= leaf_mask;
            return;
        }
    }

    if len == touches.capacity() {
        touches.reserve(1);
    }

    // SAFETY: capacity was checked above. PageTouch is Copy and has no drop glue.
    unsafe {
        touches.as_mut_ptr().add(len).write(PageTouch {
            page_id,
            _padding: 0,
            leaf_mask,
        });
        touches.set_len(len + 1);
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::metering::PAGE_MASK_LEAF_BITS_U32;

    use super::*;
    use crate::{arch::MEMORY_BLOCK_BYTES, utils::test_system_config};

    #[test]
    fn test_range_insertion_matches_explicit_leaves() {
        let system_config = test_system_config();
        let mut range_ctx = MemoryCtx::new(&system_config);
        let mut explicit_ctx = MemoryCtx::new(&system_config);

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

    #[test]
    fn exact_load_store_span_matches_aligned_memory_blocks() {
        let system_config = test_system_config();
        let ctx = MemoryCtx::new(&system_config);
        let block_size = MEMORY_BLOCK_BYTES as u32;

        for width in [1, 2, 4, 8] {
            for ptr in 0..2 * (1 << BYTE_PTRS_PER_LEAF_BITS) {
                let block_ptr = ptr / block_size * block_size;
                let block_span = if ptr - block_ptr + width > block_size {
                    2 * block_size
                } else {
                    block_size
                };

                assert_eq!(
                    ctx.leaf_id_range(RV64_MEMORY_AS, ptr, width),
                    ctx.leaf_id_range(RV64_MEMORY_AS, block_ptr, block_span),
                    "ptr={ptr}, width={width}"
                );
            }
        }
    }

    #[test]
    fn test_tentative_apply_does_not_commit_occupancy() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let mut trace_heights = vec![0; 6];

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.apply_height_updates(&mut trace_heights);

        let page_id = ((2 - ADDR_SPACE_OFFSET) << ctx.memory_dimensions.address_height) as usize
            >> PAGE_MASK_LEAF_BITS;
        assert_eq!(ctx.baseline_memory.page_mask(page_id), 0);
    }

    #[test]
    fn test_address_spaces_map_to_distinct_pages() {
        let system_config = test_system_config();
        let ctx = MemoryCtx::new(&system_config);
        let memory_page = ctx.leaf_id_range(RV64_MEMORY_AS, 0, 1).0 >> PAGE_MASK_LEAF_BITS;
        let public_values_page = ctx.leaf_id_range(PUBLIC_VALUES_AS, 0, 1).0 >> PAGE_MASK_LEAF_BITS;
        let deferral_page = ctx.leaf_id_range(DEFERRAL_AS, 0, 1).0 >> PAGE_MASK_LEAF_BITS;

        assert_ne!(memory_page, public_values_page);
        assert_ne!(memory_page, deferral_page);
        assert_ne!(public_values_page, deferral_page);
    }

    #[test]
    fn test_first_touches_poseidon_rows_dedup_by_default_bucket() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = (U16_CELL_SIZE * VM_DIGEST_WIDTH) as u32;

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.update_boundary_merkle_heights(2, second_leaf_ptr, 1);

        let mut trace_heights = vec![0; 6];
        ctx.apply_height_updates(&mut trace_heights);

        let poseidon2_idx = trace_heights.len() - 2;
        assert_eq!(trace_heights[BOUNDARY_AIR_ID], 4);
        assert_eq!(trace_heights[MERKLE_AIR_ID], 2 * height);
        assert_eq!(trace_heights[poseidon2_idx], 2 * height + 3);
    }

    #[test]
    fn test_default_poseidon_rows_dedup_across_checkpoints() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let mut trace_heights = vec![0; 6];

        ctx.update_boundary_merkle_heights(RV64_MEMORY_AS, 0, 1);
        ctx.apply_height_updates(&mut trace_heights);
        ctx.update_checkpoint();

        let boundary_before = trace_heights[BOUNDARY_AIR_ID];
        let merkle_before = trace_heights[MERKLE_AIR_ID];
        let poseidon2_idx = trace_heights.len() - 2;
        let poseidon_before = trace_heights[poseidon2_idx];
        let next_page_ptr = ((1 << PAGE_MASK_LEAF_BITS) * U16_CELL_SIZE * VM_DIGEST_WIDTH) as u32;

        ctx.update_boundary_merkle_heights(RV64_MEMORY_AS, next_page_ptr, 1);
        ctx.apply_height_updates(&mut trace_heights);

        assert_eq!(trace_heights[BOUNDARY_AIR_ID] - boundary_before, 2);
        assert_eq!(
            trace_heights[MERKLE_AIR_ID] - merkle_before,
            2 * PAGE_MASK_LEAF_BITS_U32
        );
        assert_eq!(
            trace_heights[poseidon2_idx] - poseidon_before,
            1 + PAGE_MASK_LEAF_BITS_U32
        );
    }

    #[test]
    fn test_initial_zero_bytes_do_not_seed_occupancy() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let height = ctx.memory_dimensions.overall_height() as u32;
        ctx.seed_initial_memory(&SparseMemoryImage::from([((2, 0), 0)]));
        let second_leaf_ptr = (U16_CELL_SIZE * VM_DIGEST_WIDTH) as u32;

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.update_boundary_merkle_heights(2, second_leaf_ptr, 1);

        let mut trace_heights = vec![0; 6];
        ctx.apply_height_updates(&mut trace_heights);

        let poseidon2_idx = trace_heights.len() - 2;
        assert_eq!(trace_heights[poseidon2_idx], 2 * height + 3);
    }

    #[test]
    fn test_initial_nonzero_bytes_seed_occupancy() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = (U16_CELL_SIZE * VM_DIGEST_WIDTH) as u32;
        ctx.seed_initial_memory(&SparseMemoryImage::from([
            ((2, 0), 1),
            ((2, second_leaf_ptr), 1),
        ]));

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.update_boundary_merkle_heights(2, second_leaf_ptr, 1);

        let mut trace_heights = vec![0; 6];
        ctx.apply_height_updates(&mut trace_heights);

        let poseidon2_idx = trace_heights.len() - 2;
        assert_eq!(trace_heights[poseidon2_idx], 2 * height + 4);
    }

    #[test]
    fn test_initial_deferral_bytes_seed_occupancy() {
        let system_config = test_system_config();
        let mut ctx = MemoryCtx::new(&system_config);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = VM_DIGEST_WIDTH as u32;
        let second_leaf_byte_ptr = second_leaf_ptr * FIELD_ELEMENT_BYTES;
        ctx.seed_initial_memory(&SparseMemoryImage::from([
            ((DEFERRAL_AS, 0), 1),
            ((DEFERRAL_AS, second_leaf_byte_ptr), 1),
        ]));

        ctx.update_boundary_merkle_heights(DEFERRAL_AS, 0, 1);
        ctx.update_boundary_merkle_heights(DEFERRAL_AS, second_leaf_ptr, 1);

        let mut trace_heights = vec![0; 6];
        ctx.apply_height_updates(&mut trace_heights);

        let poseidon2_idx = trace_heights.len() - 2;
        assert_eq!(trace_heights[poseidon2_idx], 2 * height + 4);
    }
}
