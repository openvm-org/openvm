use std::mem::size_of;

use openvm_instructions::{
    exe::SparseMemoryImage,
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    DEFERRAL_AS, DIGEST_WIDTH, MEMORY_PAGE_BITS,
};

pub use super::memory_tracker::PageAccess;
use super::memory_tracker::{
    leaf_mask_range, GlobalFirstTouchCounts, GlobalMemoryTracker, SegmentMemoryTracker,
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
const FIELD_ELEMENT_BYTES: u32 = size_of::<u32>() as u32;

#[derive(Clone, Debug)]
pub struct MemoryCtx {
    /// Memory tree dimensions used to map address-space ranges into global leaf ids.
    memory_dimensions: MemoryDimensions,
    /// Memory leaves and nodes already counted in the current segment.
    segment_memory: SegmentMemoryTracker,
    /// Memory leaves and nodes present in the global baseline at the last checkpoint.
    global_memory: GlobalMemoryTracker,
    /// Page masks recorded since the last safe checkpoint by the normal metered path.
    pub page_indices_since_checkpoint: Vec<PageAccess>,
    /// Length mirror used by generated metered code that appends through the raw buffer.
    pub page_indices_since_checkpoint_len: usize,
    /// Number of checkpoint-buffer entries already reflected in `trace_heights`.
    page_indices_applied_len: usize,
    /// New segment leaf masks queued for the next checkpoint. Segment
    /// replay clears this queue and recomputes it from `page_indices_since_checkpoint`.
    pending_global_updates: Vec<PageAccess>,
    /// New segment leaves accumulated by direct page-buffer application.
    pending_segment_leaves: u32,
    /// New segment Merkle nodes accumulated by direct page-buffer application.
    pending_segment_merkle_nodes: u32,
    /// Global first touches paired with the pending height deltas.
    pending_global_first_touches: GlobalFirstTouchCounts,
}

impl MemoryCtx {
    pub fn new(config: &SystemConfig, segment_check_insns: u64) -> Self {
        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        let upper_height = merkle_height.saturating_sub(MEMORY_PAGE_BITS);
        let checkpoint_capacity = Self::initial_checkpoint_capacity(segment_check_insns);

        Self {
            memory_dimensions,
            segment_memory: SegmentMemoryTracker::new(upper_height),
            global_memory: GlobalMemoryTracker::new(upper_height),
            page_indices_since_checkpoint: Vec::with_capacity(checkpoint_capacity),
            page_indices_since_checkpoint_len: 0,
            page_indices_applied_len: 0,
            pending_global_updates: Vec::with_capacity(checkpoint_capacity),
            pending_segment_leaves: 0,
            pending_segment_merkle_nodes: 0,
            pending_global_first_touches: GlobalFirstTouchCounts::default(),
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
        let start_page_id = start_leaf_id >> MEMORY_PAGE_BITS;
        let end_page_id = ((end_leaf_id - 1) >> MEMORY_PAGE_BITS) + 1;

        for page_id in start_page_id..end_page_id {
            let page_start = page_id << MEMORY_PAGE_BITS;
            let page_end = page_start + (1 << MEMORY_PAGE_BITS);
            let start = start_leaf_id.max(page_start) - page_start;
            let end = end_leaf_id.min(page_end) - page_start;
            let leaf_mask = leaf_mask_range(start, end);
            self.global_memory
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
        let start_page_id = start_leaf_id >> MEMORY_PAGE_BITS;

        if num_leaves == 1 {
            push_page_access(
                &mut self.page_indices_since_checkpoint,
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
            push_page_access(&mut self.page_indices_since_checkpoint, page_id, leaf_mask);
        }
        self.page_indices_since_checkpoint_len = self.page_indices_since_checkpoint.len();
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
            let delta =
                self.segment_memory
                    .insert(page_id as usize, access.leaf_mask, &self.global_memory);
            if delta.new_segment_leaf_mask != 0 {
                self.pending_segment_leaves += delta.segment_leaves;
                self.pending_segment_merkle_nodes += delta.segment_merkle_nodes;
                push_page_access(
                    &mut self.pending_global_updates,
                    page_id,
                    delta.new_segment_leaf_mask,
                );
                self.pending_global_first_touches
                    .add(delta.global_first_touches);
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
        self.pending_global_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_global_first_touches = GlobalFirstTouchCounts::default();

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
        self.pending_global_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_global_first_touches = GlobalFirstTouchCounts::default();

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

    /// Adds pending segment updates to the global memory baseline.
    #[inline(always)]
    pub(crate) fn update_checkpoint(&mut self) {
        self.global_memory
            .add_page_accesses(&self.pending_global_updates);
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_global_updates.clear();
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_global_first_touches = GlobalFirstTouchCounts::default();
    }

    /// Applies memory height deltas recorded since the last checkpoint.
    ///
    /// - BOUNDARY_AIR: `2 * segment_leaves` rows
    /// - MERKLE_AIR:   `2 * segment_merkle_nodes` rows
    /// - Poseidon2:    final-side hashes plus initial-side hashes
    ///
    /// A global first touch means the initial-side value is canonical default.
    /// Default leaves share one Poseidon2 row. Default internal nodes share one
    /// Poseidon2 row per Merkle height.
    #[inline(always)]
    pub(crate) fn apply_height_updates(&mut self, trace_heights: &mut [u32]) {
        let mut leaves = self.pending_segment_leaves;
        let mut merkle_nodes = self.pending_segment_merkle_nodes;
        let mut global_first_touches = self.pending_global_first_touches;
        self.pending_segment_leaves = 0;
        self.pending_segment_merkle_nodes = 0;
        self.pending_global_first_touches = GlobalFirstTouchCounts::default();

        let len = self.page_indices_since_checkpoint.len();
        let ptr = self.page_indices_since_checkpoint.as_ptr();
        for i in self.page_indices_applied_len..len {
            // SAFETY: i is bounded by page_indices_since_checkpoint.len().
            let access = unsafe { *ptr.add(i) };
            let delta = self.segment_memory.insert(
                access.page_id as usize,
                access.leaf_mask,
                &self.global_memory,
            );
            if delta.new_segment_leaf_mask != 0 {
                leaves += delta.segment_leaves;
                merkle_nodes += delta.segment_merkle_nodes;
                push_page_access(
                    &mut self.pending_global_updates,
                    access.page_id,
                    delta.new_segment_leaf_mask,
                );
                global_first_touches.add(delta.global_first_touches);
            }
        }
        self.page_indices_applied_len = len;

        if leaves == 0 && merkle_nodes == 0 {
            return;
        }

        debug_assert!(trace_heights.len() >= 2);
        let poseidon2_idx = trace_heights.len() - 2;
        let initial_default_poseidon_rows = global_first_touches.estimated_default_poseidon_rows();
        let initial_nondefault_poseidon_rows = (leaves + merkle_nodes)
            .saturating_sub(global_first_touches.leaves + global_first_touches.merkle_nodes);
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
fn push_page_access(accesses: &mut Vec<PageAccess>, page_id: u32, leaf_mask: u64) {
    debug_assert!(leaf_mask != 0);
    let len = accesses.len();
    if len != 0 {
        // SAFETY: len is non-zero, so len - 1 is in bounds.
        let prev = unsafe { accesses.get_unchecked_mut(len - 1) };
        if prev.page_id == page_id {
            prev.leaf_mask |= leaf_mask;
            return;
        }
    }

    if len == accesses.capacity() {
        accesses.reserve(1);
    }

    // SAFETY: capacity was checked above. PageAccess is Copy and has no drop glue.
    unsafe {
        accesses
            .as_mut_ptr()
            .add(len)
            .write(PageAccess { page_id, leaf_mask });
        accesses.set_len(len + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_tentative_apply_does_not_commit_occupancy() {
        let system_config = crate::utils::test_system_config();
        let mut ctx = MemoryCtx::new(&system_config, 1);
        let mut trace_heights = vec![0; 6];

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.apply_height_updates(&mut trace_heights);

        let page_id = ((2 - ADDR_SPACE_OFFSET) << ctx.memory_dimensions.address_height) as usize
            >> MEMORY_PAGE_BITS;
        assert_eq!(ctx.global_memory.page_mask(page_id), 0);
    }

    #[test]
    fn test_global_first_touches_poseidon_rows_dedup_by_default_bucket() {
        let system_config = crate::utils::test_system_config();
        let mut ctx = MemoryCtx::new(&system_config, 1);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = (U16_CELL_SIZE * DIGEST_WIDTH) as u32;

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
    fn test_initial_zero_bytes_do_not_seed_occupancy() {
        let system_config = crate::utils::test_system_config();
        let mut ctx = MemoryCtx::new(&system_config, 1);
        let height = ctx.memory_dimensions.overall_height() as u32;
        ctx.seed_initial_memory(&SparseMemoryImage::from([((2, 0), 0)]));
        let second_leaf_ptr = (U16_CELL_SIZE * DIGEST_WIDTH) as u32;

        ctx.update_boundary_merkle_heights(2, 0, 1);
        ctx.update_boundary_merkle_heights(2, second_leaf_ptr, 1);

        let mut trace_heights = vec![0; 6];
        ctx.apply_height_updates(&mut trace_heights);

        let poseidon2_idx = trace_heights.len() - 2;
        assert_eq!(trace_heights[poseidon2_idx], 2 * height + 3);
    }

    #[test]
    fn test_initial_nonzero_bytes_seed_occupancy() {
        let system_config = crate::utils::test_system_config();
        let mut ctx = MemoryCtx::new(&system_config, 1);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = (U16_CELL_SIZE * DIGEST_WIDTH) as u32;
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
        let system_config = crate::utils::test_system_config();
        let mut ctx = MemoryCtx::new(&system_config, 1);
        let height = ctx.memory_dimensions.overall_height() as u32;
        let second_leaf_ptr = DIGEST_WIDTH as u32;
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
