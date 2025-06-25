use openvm_instructions::riscv::RV32_IMM_AS;
use serde::{Deserialize, Serialize};

use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, PUBLIC_VALUES_AIR_ID},
    system::memory::{dimensions::MemoryDimensions, CHUNK},
};

#[derive(Debug)]
pub struct BitSet {
    words: Box<[u64]>,
}

impl BitSet {
    pub fn new(size_bits: usize) -> Self {
        let num_words = 1 << size_bits.saturating_sub(6);
        Self {
            words: vec![0; num_words].into_boxed_slice(),
        }
    }

    pub fn insert(&mut self, index: usize) -> bool {
        let word_index = index / 64;
        let bit_index = index % 64;
        let mask = 1u64 << bit_index;

        let was_set = (self.words[word_index] & mask) != 0;
        self.words[word_index] |= mask;
        !was_set
    }

    pub fn clear(&mut self) {
        for item in self.words.iter_mut() {
            *item = 0;
        }
    }
}

#[derive(Debug)]
pub struct MeteredCtx<const PAGE_BITS: usize = 12> {
    pub trace_heights: Vec<u32>,
    pub is_trace_height_constant: Vec<bool>,

    pub page_indices: BitSet,

    pub instret_last_segment_check: u64,
    pub segments: Vec<Segment>,

    memory_dimensions: MemoryDimensions,
    as_byte_alignment_bits: Vec<u8>,
    boundary_idx: usize,
    merkle_tree_index: Option<usize>,
    adapter_offset: usize,
    chunk: u32,
    chunk_bits: u32,
}

impl<const PAGE_BITS: usize> MeteredCtx<PAGE_BITS> {
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
        let merkle_height = memory_dimensions.overall_height();

        Self {
            trace_heights: vec![0; num_traces],
            is_trace_height_constant: vec![false; num_traces],
            page_indices: BitSet::new(merkle_height.saturating_sub(PAGE_BITS)),
            instret_last_segment_check: 0,
            segments: Vec::new(),
            as_byte_alignment_bits,
            boundary_idx,
            merkle_tree_index,
            adapter_offset,
            chunk,
            chunk_bits,
            memory_dimensions,
        }
    }

    fn update_boundary_merkle_heights(&mut self, address_space: u32, ptr: u32, size: u32) {
        let num_blocks = (size + self.chunk - 1) >> self.chunk_bits;
        let mut addr = ptr;
        for _ in 0..num_blocks {
            let block_id = addr >> self.chunk_bits;
            let index = if self.chunk == 1 {
                // Volatile
                block_id
            } else {
                self.memory_dimensions
                    .label_to_index((address_space, block_id)) as u32
            } as usize;

            if self.page_indices.insert(index >> PAGE_BITS) {
                // On page fault, assume we add all leaves in a page
                let leaves = 1 << PAGE_BITS;
                self.trace_heights[self.boundary_idx] += leaves;

                if let Some(merkle_tree_idx) = self.merkle_tree_index {
                    let poseidon2_idx = self.trace_heights.len() - 2;
                    self.trace_heights[poseidon2_idx] += leaves * 2;

                    let merkle_height = self.memory_dimensions.overall_height();
                    let nodes = (((1 << PAGE_BITS) - 1) + (merkle_height - PAGE_BITS)) as u32;
                    self.trace_heights[poseidon2_idx] += nodes * 2;
                    self.trace_heights[merkle_tree_idx] += nodes * 2;
                }

                // At finalize, we'll need to read it in chunk-sized units for the merkle chip
                self.update_adapter_heights_batch(address_space, self.chunk_bits, leaves);
            }

            addr = addr.wrapping_add(self.chunk);
        }
    }

    fn update_adapter_heights(&mut self, address_space: u32, size_bits: u32) {
        self.update_adapter_heights_batch(address_space, size_bits, 1);
    }

    fn update_adapter_heights_batch(&mut self, address_space: u32, size_bits: u32, num: u32) {
        let align_bits = self.as_byte_alignment_bits[address_space as usize];
        debug_assert!(
            align_bits as u32 <= size_bits,
            "align_bits ({}) must be <= size_bits ({})",
            align_bits,
            size_bits
        );
        for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
            let adapter_idx = self.adapter_offset + adapter_bits as usize - 1;
            self.trace_heights[adapter_idx] += num << (size_bits - adapter_bits + 1);
        }
    }
}

impl<const PAGE_BITS: usize> E1E2ExecutionCtx for MeteredCtx<PAGE_BITS> {
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

#[derive(derive_new::new, Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}
