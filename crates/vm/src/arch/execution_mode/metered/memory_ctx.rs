use std::mem::size_of;

use openvm_instructions::{
    exe::SparseMemoryImage,
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    DEFERRAL_AS, DIGEST_WIDTH, MEMORY_PAGE_BITS,
};

use super::page_tracker::{
    leaf_mask_range, CommittedMemoryOccupancyTracker, DefaultOldAccounting,
    SegmentMemoryPageTracker,
};
pub use super::page_tracker::PageAccess;
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
#[derive(Clone, Debug)]
pub struct MemoryCtx {
    memory_dimensions: MemoryDimensions,
    /// Segment-local occupancy. Cleared at segment boundaries; it prevents
    /// charging the same leaf or Merkle node twice inside one segment.
    page_tracker: SegmentMemoryPageTracker,
    /// Committed occupancy at the last safe checkpoint, seeded with nonzero
    /// initial memory. This tells us whether an old value is canonical default.
    occupancy_tracker: CommittedMemoryOccupancyTracker,
    pub page_indices_since_checkpoint: Vec<PageAccess>,
    pub page_indices_since_checkpoint_len: usize,
    page_indices_applied_len: usize,
    /// New leaves already reflected in `trace_heights` and queued for the next
    /// checkpoint commit. Segment replay clears this queue and recomputes
    /// heights from `page_indices_since_checkpoint`.
    pending_occupancy_updates: Vec<PageAccess>,
    pending_leaves: u32,
    pending_merkle_nodes: u32,
    pending_default_old: DefaultOldAccounting,
}

impl MemoryCtx {
    pub fn new(config: &SystemConfig, segment_check_insns: u64) -> Self {
        let memory_dimensions = config.memory_config.memory_dimensions();
        let merkle_height = memory_dimensions.overall_height();

        let upper_height = merkle_height.saturating_sub(MEMORY_PAGE_BITS);
        let checkpoint_capacity = Self::initial_checkpoint_capacity(segment_check_insns);

        Self {
            memory_dimensions,
            page_tracker: SegmentMemoryPageTracker::new(upper_height),
            occupancy_tracker: CommittedMemoryOccupancyTracker::new(upper_height),
            page_indices_since_checkpoint: Vec::with_capacity(checkpoint_capacity),
            page_indices_since_checkpoint_len: 0,
            page_indices_applied_len: 0,
            pending_occupancy_updates: Vec::with_capacity(checkpoint_capacity),
            pending_leaves: 0,
            pending_merkle_nodes: 0,
            pending_default_old: DefaultOldAccounting::default(),
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
            self.occupancy_tracker
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
            let delta = self.page_tracker.insert(
                page_id as usize,
                access.leaf_mask,
                &self.occupancy_tracker,
            );
            if delta.newly_charged_leaf_mask != 0 {
                self.pending_leaves += delta.new_leaves;
                self.pending_merkle_nodes += delta.new_merkle_nodes;
                push_page_access(
                    &mut self.pending_occupancy_updates,
                    page_id,
                    delta.newly_charged_leaf_mask,
                );
                self.pending_default_old.add(delta.default_old);
            }
        }
    }

    #[cfg(feature = "rvr")]
    #[inline(always)]
    pub(crate) fn reset_segment_without_replay(&mut self, trace_heights: &mut [u32]) {
        self.page_tracker.clear();
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_occupancy_updates.clear();
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;
        self.pending_default_old = DefaultOldAccounting::default();

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
        self.pending_occupancy_updates.clear();
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;
        self.pending_default_old = DefaultOldAccounting::default();

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
        self.occupancy_tracker
            .commit_page_accesses(&self.pending_occupancy_updates);
        self.page_indices_since_checkpoint.clear();
        self.page_indices_since_checkpoint_len = 0;
        self.page_indices_applied_len = 0;
        self.pending_occupancy_updates.clear();
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;
        self.pending_default_old = DefaultOldAccounting::default();
    }

    /// Applies memory height deltas recorded since the last checkpoint.
    ///
    /// - BOUNDARY_AIR: `2 * new_leaves` rows
    /// - MERKLE_AIR:   `2 * new_merkle_nodes` rows
    /// - Poseidon2:    final-side hashes plus old-side hashes
    ///
    /// Old-side default leaves share one Poseidon2 input. Old-side default
    /// internal nodes share one input per Merkle height.
    #[inline(always)]
    pub(crate) fn apply_height_updates(&mut self, trace_heights: &mut [u32]) {
        let mut leaves = self.pending_leaves;
        let mut merkle_nodes = self.pending_merkle_nodes;
        let mut default_old = self.pending_default_old;
        self.pending_leaves = 0;
        self.pending_merkle_nodes = 0;
        self.pending_default_old = DefaultOldAccounting::default();

        let len = self.page_indices_since_checkpoint.len();
        let ptr = self.page_indices_since_checkpoint.as_ptr();
        for i in self.page_indices_applied_len..len {
            // SAFETY: i is bounded by page_indices_since_checkpoint.len().
            let access = unsafe { *ptr.add(i) };
            let delta = self.page_tracker.insert(
                access.page_id as usize,
                access.leaf_mask,
                &self.occupancy_tracker,
            );
            if delta.newly_charged_leaf_mask != 0 {
                leaves += delta.new_leaves;
                merkle_nodes += delta.new_merkle_nodes;
                push_page_access(
                    &mut self.pending_occupancy_updates,
                    access.page_id,
                    delta.newly_charged_leaf_mask,
                );
                default_old.add(delta.default_old);
            }
        }
        self.page_indices_applied_len = len;

        if leaves == 0 && merkle_nodes == 0 {
            return;
        }

        debug_assert!(trace_heights.len() >= 2);
        let poseidon2_idx = trace_heights.len() - 2;
        let old_default_poseidon_rows = default_old.estimated_poseidon_rows();
        let old_nondefault_poseidon_rows = (leaves + merkle_nodes)
            .saturating_sub(default_old.default_leaves + default_old.default_merkle_nodes);
        let poseidon2_rows =
            leaves + merkle_nodes + old_nondefault_poseidon_rows + old_default_poseidon_rows;
        // SAFETY: BOUNDARY_AIR_ID, MERKLE_AIR_ID, and poseidon2_idx are all within bounds
        unsafe {
            *trace_heights.get_unchecked_mut(BOUNDARY_AIR_ID) += leaves * 2;
            *trace_heights.get_unchecked_mut(poseidon2_idx) += poseidon2_rows;
            // Merkle AIR: 2 rows per internal node (init + final tree)
            *trace_heights.get_unchecked_mut(MERKLE_AIR_ID) += merkle_nodes * 2;
        }
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

        let page_id =
            ((2 - ADDR_SPACE_OFFSET) << ctx.memory_dimensions.address_height) as usize >> MEMORY_PAGE_BITS;
        assert_eq!(ctx.occupancy_tracker.page_mask(page_id), 0);
    }

    #[test]
    fn test_default_old_poseidon_rows_dedup_by_default_bucket() {
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
