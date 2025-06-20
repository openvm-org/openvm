use openvm_instructions::riscv::RV32_IMM_AS;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, PUBLIC_VALUES_AIR_ID},
    system::memory::{dimensions::MemoryDimensions, CHUNK},
};

// TODO(ayush): can segmentation also be triggered by timestamp overflow? should that be tracked?
#[derive(Debug)]
pub struct MeteredCtx {
    pub trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,

    // Indices of leaf nodes in the memory merkle tree
    pub leaf_indices: FxHashSet<u64>,

    pub instret_last_segment_check: u64,
    pub segments: Vec<Segment>,

    memory_dimensions: MemoryDimensions,
    as_byte_alignment_bits: Vec<u8>,
    boundary_idx: usize,
    merkle_tree_index: Option<usize>,
    adapter_offset: usize,
    chunk: u32,
    chunk_bits: u32,
    memory_merkle_height: u32,
}

impl MeteredCtx {
    pub fn new(
        num_traces: usize,
        continuations_enabled: bool,
        as_byte_alignment_bits: Vec<u8>,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        let boundary_idx = if continuations_enabled {
            PUBLIC_VALUES_AIR_ID
        } else {
            PUBLIC_VALUES_AIR_ID + 1
        };

        let merkle_tree_index = if continuations_enabled {
            Some(boundary_idx + 1)
        } else {
            None
        };

        let adapter_offset = if continuations_enabled {
            boundary_idx + 2
        } else {
            boundary_idx
        };

        let chunk = if continuations_enabled {
            // Persistent memory uses CHUNK-sized blocks
            CHUNK as u32
        } else {
            // Volatile memory uses single units
            1
        };

        let chunk_bits = chunk.ilog2();
        let memory_merkle_height = memory_dimensions.overall_height() as u32;

        Self {
            trace_heights: vec![0; num_traces],
            is_trace_height_constant: vec![false; num_traces],
            leaf_indices: FxHashSet::default(),
            instret_last_segment_check: 0,
            segments: Vec::new(),
            memory_dimensions,
            as_byte_alignment_bits,
            boundary_idx,
            merkle_tree_index,
            adapter_offset,
            chunk,
            chunk_bits,
            memory_merkle_height,
        }
    }
}

impl MeteredCtx {
    fn update_boundary_merkle_heights(&mut self, address_space: u32, ptr: u32, size: u32) {
        let num_blocks = (size + self.chunk - 1) >> self.chunk_bits;
        for i in 0..num_blocks {
            let addr = ptr.wrapping_add(i * self.chunk);
            let block_id = addr >> self.chunk_bits;
            let leaf_id = self
                .memory_dimensions
                .label_to_index((address_space, block_id));

            if self.leaf_indices.insert(leaf_id) {
                let poseidon2_idx = self.trace_heights.len() - 2;

                self.trace_heights[self.boundary_idx] += 1;
                self.trace_heights[poseidon2_idx] += 2;

                if let Some(merkle_tree_idx) = self.merkle_tree_index {
                    self.trace_heights[merkle_tree_idx] += self.memory_merkle_height * 2;
                    self.trace_heights[poseidon2_idx] += self.memory_merkle_height * 2;
                }

                // At finalize, we'll need to read it in chunk-sized units for the merkle chip
                self.update_adapter_heights(address_space, self.chunk_bits);
            }
        }
    }

    fn update_adapter_heights(&mut self, address_space: u32, size_bits: u32) {
        let align_bits = self.as_byte_alignment_bits[address_space as usize];
        debug_assert!(
            align_bits as u32 <= size_bits,
            "align_bits ({}) must be <= size_bits ({})",
            align_bits,
            size_bits
        );
        for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
            self.trace_heights[self.adapter_offset + adapter_bits as usize - 1] +=
                1 << (size_bits - adapter_bits + 1);
        }
    }
}

impl E1E2ExecutionCtx for MeteredCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(
            size.is_power_of_two(),
            "size must be a power of 2, got {}",
            size
        );

        // Handle access adapter updates
        let size_bits = size.ilog2();
        self.update_adapter_heights(address_space, size_bits);

        // Handle merkle tree updates
        // TODO(ayush): use a looser upper bound
        // see if this can be approximated by total number of reads/writes for AS != register
        self.update_boundary_merkle_heights(address_space, ptr, size);
    }
}

#[derive(derive_new::new, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}
