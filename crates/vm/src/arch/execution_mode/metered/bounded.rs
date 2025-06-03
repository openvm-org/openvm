use openvm_instructions::riscv::RV32_IMM_AS;

use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, PUBLIC_VALUES_AIR_ID},
    system::memory::{dimensions::MemoryDimensions, CHUNK, CHUNK_BITS},
};

// TODO(ayush): can segmentation also be triggered by timestamp overflow? should that be tracked?
#[derive(Debug)]
pub struct MeteredCtxBounded {
    pub trace_heights: Vec<u32>,

    continuations_enabled: bool,
    num_access_adapters: u8,
    // TODO(ayush): take alignment into account for access adapters
    #[allow(dead_code)]
    as_byte_alignment_bits: Vec<u8>,
    pub memory_dimensions: MemoryDimensions,

    // Indices of leaf nodes in the memory merkle tree
    pub leaf_indices: Vec<u64>,
    pub clk_last_segment_check: u64,
    pub segments: Vec<Segment>,
}

impl MeteredCtxBounded {
    pub fn new(
        num_traces: usize,
        continuations_enabled: bool,
        num_access_adapters: u8,
        as_byte_alignment_bits: Vec<u8>,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        Self {
            trace_heights: vec![0; num_traces],
            continuations_enabled,
            num_access_adapters,
            as_byte_alignment_bits,
            memory_dimensions,
            leaf_indices: Vec::new(),
            clk_last_segment_check: 0,
            segments: Vec::new(),
        }
    }
}

impl MeteredCtxBounded {
    fn update_boundary_merkle_heights(&mut self, address_space: u32, ptr: u32, size: u32) {
        let boundary_idx = if self.continuations_enabled {
            PUBLIC_VALUES_AIR_ID
        } else {
            PUBLIC_VALUES_AIR_ID + 1
        };
        let poseidon2_idx = self.trace_heights.len() - 2;

        let num_blocks = (size + CHUNK as u32 - 1) >> CHUNK_BITS;
        for i in 0..num_blocks {
            let addr = ptr.wrapping_add(i * CHUNK as u32);
            let block_id = addr >> CHUNK_BITS;
            let leaf_id = self
                .memory_dimensions
                .label_to_index((address_space, block_id));

            if let Err(insert_idx) = self.leaf_indices.binary_search(&leaf_id) {
                self.leaf_indices.insert(insert_idx, leaf_id);

                self.trace_heights[boundary_idx] += 1;
                self.trace_heights[poseidon2_idx] += 2;

                if self.continuations_enabled {
                    let pred_id = insert_idx.checked_sub(1).map(|idx| self.leaf_indices[idx]);
                    let succ_id = (insert_idx < self.leaf_indices.len() - 1)
                        .then(|| self.leaf_indices[insert_idx + 1]);
                    let height_change = calculate_merkle_node_updates(
                        leaf_id,
                        pred_id,
                        succ_id,
                        self.memory_dimensions.overall_height() as u32,
                    );
                    self.trace_heights[boundary_idx + 1] += height_change * 2;
                    self.trace_heights[poseidon2_idx] += height_change * 2;
                }
            }
        }
    }

    fn update_adapter_heights_batch(&mut self, size: u32, num: u32) {
        let adapter_offset = if self.continuations_enabled {
            PUBLIC_VALUES_AIR_ID + 2
        } else {
            PUBLIC_VALUES_AIR_ID + 1
        };

        apply_adapter_updates_batch(size, num, &mut self.trace_heights[adapter_offset..]);
    }

    fn update_adapter_heights(&mut self, size: u32) {
        self.update_adapter_heights_batch(size, 1);
    }

    pub fn finalize_access_adapter_heights(&mut self) {
        self.update_adapter_heights_batch(CHUNK as u32, self.leaf_indices.len() as u32);
    }

    pub fn trace_heights_if_finalized(&mut self) -> Vec<u32> {
        let num_leaves = self.leaf_indices.len() as u32;
        let mut access_adapter_updates = vec![0; self.num_access_adapters as usize];
        apply_adapter_updates_batch(CHUNK as u32, num_leaves, &mut access_adapter_updates);

        let adapter_offset = if self.continuations_enabled {
            PUBLIC_VALUES_AIR_ID + 2
        } else {
            PUBLIC_VALUES_AIR_ID + 1
        };
        self.trace_heights
            .iter()
            .enumerate()
            .map(|(i, &height)| {
                if i >= adapter_offset && i < adapter_offset + access_adapter_updates.len() {
                    height + access_adapter_updates[i - adapter_offset]
                } else {
                    height
                }
            })
            .collect()
    }
}

impl E1E2ExecutionCtx for MeteredCtxBounded {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(size.is_power_of_two(), "size must be a power of 2");

        // Handle access adapter updates
        self.update_adapter_heights(size);

        // Handle merkle tree updates
        // TODO(ayush): use a looser upper bound
        // see if this can be approximated by total number of reads/writes for AS != register
        self.update_boundary_merkle_heights(address_space, ptr, size);
    }
}

fn apply_adapter_updates_batch(size: u32, num: u32, trace_heights: &mut [u32]) {
    let size_bits = size.ilog2();
    for adapter_bits in (3..=size_bits).rev() {
        trace_heights[adapter_bits as usize - 1] += num << (size_bits - adapter_bits + 1);
    }
}

fn calculate_merkle_node_updates(
    leaf_id: u64,
    pred_id: Option<u64>,
    succ_id: Option<u64>,
    height: u32,
) -> u32 {
    // First node requires height many updates
    if pred_id.is_none() && succ_id.is_none() {
        return height;
    }

    // Calculate the difference in divergence
    let mut diff = 0;

    // Add new divergences between pred and leaf_index
    if let Some(p) = pred_id {
        let new_divergence = (p ^ leaf_id).ilog2();
        diff += new_divergence;
    }

    // Add new divergences between leaf_index and succ
    if let Some(s) = succ_id {
        let new_divergence = (leaf_id ^ s).ilog2();
        diff += new_divergence;
    }

    // Remove old divergence between pred and succ if both existed
    if let (Some(p), Some(s)) = (pred_id, succ_id) {
        let old_divergence = (p ^ s).ilog2();
        diff -= old_divergence;
    }

    diff
}

#[derive(derive_new::new, Debug)]
pub struct Segment {
    pub clk_start: u64,
    pub num_cycles: u64,
    pub trace_heights: Vec<u32>,
}
