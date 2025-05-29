use std::collections::BTreeMap;

use openvm_instructions::riscv::RV32_IMM_AS;

use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, PUBLIC_VALUES_AIR_ID},
    system::memory::{dimensions::MemoryDimensions, CHUNK, CHUNK_BITS},
};

// TODO(ayush): can segmentation also be triggered by timestamp overflow? should that be tracked?
#[derive(Debug)]
pub struct MeteredCtxExact {
    pub trace_heights: Vec<u32>,

    continuations_enabled: bool,
    num_access_adapters: u8,
    as_byte_alignment_bits: Vec<u8>,
    pub memory_dimensions: MemoryDimensions,

    // Map from (addr_space, addr) -> (size, offset)
    pub last_memory_access: BTreeMap<(u8, u32), (u8, u8)>,
    // Indices of leaf nodes in the memory merkle tree
    pub leaf_indices: Vec<u64>,
}

impl MeteredCtxExact {
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
            last_memory_access: BTreeMap::new(),
            leaf_indices: Vec::new(),
        }
    }
}

impl MeteredCtxExact {
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
                // NOTE: this is an upper bound since poseidon chip removes duplicates
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

    // TODO(ayush): fix this for native
    #[allow(clippy::type_complexity)]
    fn calculate_splits_and_merges(
        &self,
        address_space: u32,
        ptr: u32,
        size: u32,
    ) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        // Skip adapters if this is a repeated access to the same location with same size
        let last_access = self.last_memory_access.get(&(address_space as u8, ptr));
        if matches!(last_access, Some(&(last_access_size, 0)) if size == last_access_size as u32) {
            return (vec![], vec![]);
        }

        // Go to the start of block
        let mut ptr_start = ptr;
        if let Some(&(_, last_access_offset)) = last_access {
            ptr_start = ptr.wrapping_sub(last_access_offset as u32);
        }

        let align_bits = self.as_byte_alignment_bits[address_space as usize] as usize;
        let align = 1 << align_bits;

        // Split intersecting blocks to align bytes
        let mut curr_block = ptr_start >> align_bits;
        let end_block = curr_block + (size >> align_bits);
        let mut splits = vec![];
        while curr_block < end_block {
            let curr_block_size = if let Some(&(last_access_size, _)) = self
                .last_memory_access
                .get(&(address_space as u8, curr_block.wrapping_mul(align as u32)))
            {
                last_access_size as u32
            } else {
                // Initial memory access only happens at CHUNK boundary
                let chunk_ratio = 1 << (CHUNK_BITS - align_bits);
                let chunk_offset = curr_block & (chunk_ratio - 1);
                curr_block -= chunk_offset;
                CHUNK as u32
            };

            if curr_block_size > align as u32 {
                let curr_ptr = curr_block.wrapping_mul(align as u32);
                splits.push((curr_ptr, curr_block_size));
            }

            curr_block += curr_block_size >> align_bits;
        }
        // Merge added blocks from align to size bytes
        let merges = vec![(ptr, size)];

        (splits, merges)
    }

    #[allow(clippy::type_complexity)]
    fn apply_adapter_updates(
        &mut self,
        addr_space: u32,
        ptr: u32,
        size: u32,
        trace_heights: &mut Option<&mut [u32]>,
        memory_updates: &mut Option<Vec<((u8, u32), Option<(u8, u8)>)>>,
    ) {
        let adapter_offset = if self.continuations_enabled {
            PUBLIC_VALUES_AIR_ID + 2
        } else {
            PUBLIC_VALUES_AIR_ID + 1
        };

        let (splits, merges) = self.calculate_splits_and_merges(addr_space, ptr, size);
        for (curr_ptr, curr_size) in splits {
            if let Some(trace_heights) = trace_heights {
                apply_single_adapter_heights_update(trace_heights, curr_size);
            } else {
                apply_single_adapter_heights_update(
                    &mut self.trace_heights[adapter_offset..],
                    curr_size,
                );
            }
            let updates = add_memory_access_split_with_return(
                &mut self.last_memory_access,
                (addr_space, curr_ptr),
                curr_size,
                self.as_byte_alignment_bits[addr_space as usize],
            );
            if let Some(memory_updates) = memory_updates {
                memory_updates.extend(&updates);
            }
        }
        for (curr_ptr, curr_size) in merges {
            if let Some(trace_heights) = trace_heights {
                apply_single_adapter_heights_update(trace_heights, curr_size);
            } else {
                apply_single_adapter_heights_update(
                    &mut self.trace_heights[adapter_offset..],
                    curr_size,
                );
            }
            let updates = add_memory_access_merge_with_return(
                &mut self.last_memory_access,
                (addr_space, curr_ptr),
                curr_size,
                self.as_byte_alignment_bits[addr_space as usize],
            );
            if let Some(memory_updates) = memory_updates {
                memory_updates.extend(updates);
            }
        }
    }

    fn update_adapter_heights(&mut self, addr_space: u32, ptr: u32, size: u32) {
        self.apply_adapter_updates(addr_space, ptr, size, &mut None, &mut None);
    }

    pub fn finalize_access_adapter_heights(&mut self) {
        let indices_to_process: Vec<_> = self
            .leaf_indices
            .iter()
            .map(|&idx| {
                let (addr_space, block_id) = self.memory_dimensions.index_to_label(idx);
                (addr_space, block_id)
            })
            .collect();
        for (addr_space, block_id) in indices_to_process {
            self.update_adapter_heights(addr_space, block_id * CHUNK as u32, CHUNK as u32);
        }
    }

    pub fn trace_heights_if_finalized(&mut self) -> Vec<u32> {
        let indices_to_process: Vec<_> = self
            .leaf_indices
            .iter()
            .map(|&idx| {
                let (addr_space, block_id) = self.memory_dimensions.index_to_label(idx);
                (addr_space, block_id)
            })
            .collect();

        let mut access_adapter_updates = vec![0; self.num_access_adapters as usize];
        let mut memory_updates = Some(vec![]);
        for (addr_space, block_id) in indices_to_process {
            let ptr = block_id * CHUNK as u32;
            self.apply_adapter_updates(
                addr_space,
                ptr,
                CHUNK as u32,
                &mut Some(&mut access_adapter_updates),
                &mut memory_updates,
            );
        }

        // Restore original memory state
        for (key, old_value) in memory_updates.unwrap().into_iter().rev() {
            match old_value {
                Some(value) => {
                    self.last_memory_access.insert(key, value);
                }
                None => {
                    self.last_memory_access.remove(&key);
                }
            }
        }

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

impl E1E2ExecutionCtx for MeteredCtxExact {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(size.is_power_of_two(), "size must be a power of 2");

        // Handle access adapter updates
        self.update_adapter_heights(address_space, ptr, size);

        // Handle merkle tree updates
        // TODO(ayush): see if this can be approximated by total number of reads/writes for AS !=
        // register
        self.update_boundary_merkle_heights(address_space, ptr, size);
    }
}

fn apply_single_adapter_heights_update(trace_heights: &mut [u32], size: u32) {
    let size_bits = size.ilog2();
    for adapter_bits in (3..=size_bits).rev() {
        trace_heights[adapter_bits as usize - 1] += 1 << (size_bits - adapter_bits);
    }
}

#[allow(clippy::type_complexity)]
fn add_memory_access(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: u32,
    align_bits: u8,
    is_split: bool,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    let align = 1 << align_bits;
    debug_assert_eq!(
        size & (align as u32 - 1),
        0,
        "Size must be a multiple of alignment"
    );

    let num_chunks = size >> align_bits;
    let mut old_values = Vec::with_capacity(num_chunks as usize);

    for i in 0..num_chunks {
        let curr_ptr = ptr.wrapping_add(i * align as u32);
        let key = (address_space as u8, curr_ptr);

        let value = if is_split {
            (align as u8, 0)
        } else {
            (size as u8, (i * align as u32) as u8)
        };

        let old_value = memory_access_map.insert(key, value);
        old_values.push((key, old_value));
    }

    old_values
}

#[allow(clippy::type_complexity)]
fn add_memory_access_split_with_return(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: u32,
    align_bits: u8,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    add_memory_access(
        memory_access_map,
        (address_space, ptr),
        size,
        align_bits,
        true,
    )
}

#[allow(clippy::type_complexity)]
fn add_memory_access_merge_with_return(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: u32,
    align_bits: u8,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    add_memory_access(
        memory_access_map,
        (address_space, ptr),
        size,
        align_bits,
        false,
    )
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
