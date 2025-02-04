pub mod public_values;

use std::{ops::Range, sync::Arc};

use openvm_stark_backend::p3_field::PrimeField32;
use MemoryNode::*;

use super::controller::dimensions::MemoryDimensions;
use crate::{
    arch::hasher::{Hasher, HasherChip},
    system::memory::MemoryImage,
};

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryNode<const CHUNK: usize, F: PrimeField32> {
    Leaf {
        values: [F; CHUNK],
    },
    NonLeaf {
        hash: [F; CHUNK],
        left: Arc<MemoryNode<CHUNK, F>>,
        right: Arc<MemoryNode<CHUNK, F>>,
    },
}

impl<const CHUNK: usize, F: PrimeField32> MemoryNode<CHUNK, F> {
    pub fn hash(&self) -> [F; CHUNK] {
        match self {
            Leaf { values: hash } => *hash,
            NonLeaf { hash, .. } => *hash,
        }
    }

    pub fn new_leaf(values: [F; CHUNK]) -> Self {
        Leaf { values }
    }

    pub fn new_nonleaf(
        left: Arc<MemoryNode<CHUNK, F>>,
        right: Arc<MemoryNode<CHUNK, F>>,
        hasher: &mut impl HasherChip<CHUNK, F>,
    ) -> Self {
        NonLeaf {
            hash: hasher.compress_and_record(&left.hash(), &right.hash()),
            left,
            right,
        }
    }

    /// Returns a tree of height `height` with all leaves set to `leaf_value`.
    pub fn construct_uniform(
        height: usize,
        leaf_value: [F; CHUNK],
        hasher: &impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Self::new_leaf(leaf_value)
        } else {
            let child = Arc::new(Self::construct_uniform(height - 1, leaf_value, hasher));
            NonLeaf {
                hash: hasher.compress(&child.hash(), &child.hash()),
                left: child.clone(),
                right: child,
            }
        }
    }

    fn from_memory(
        memory: &Vec<(u64, [F; CHUNK])>,
        lookup_range: Range<usize>,
        height: usize,
        from: u64,
        hasher: &impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            if lookup_range.is_empty() {
                MemoryNode::new_leaf(hasher.hash(&[F::ZERO; CHUNK]))
            } else {
                debug_assert_eq!(memory[lookup_range.start].0, from);
                debug_assert_eq!(lookup_range.end - lookup_range.start, 1);
                MemoryNode::new_leaf(hasher.hash(&memory[lookup_range.start].1))
            }
        } else if lookup_range.is_empty() {
            let leaf_value = hasher.hash(&[F::ZERO; CHUNK]);
            MemoryNode::construct_uniform(height, leaf_value, hasher)
        } else {
            let midpoint = from + (1 << (height - 1));
            let mid = {
                let mut left = lookup_range.start;
                let mut right = lookup_range.end;
                if memory[left].0 >= midpoint {
                    left
                } else {
                    while left + 1 < right {
                        let mid = left + (right - left) / 2;
                        if memory[mid].0 < midpoint {
                            left = mid;
                        } else {
                            right = mid;
                        }
                    }
                    right
                }
            };
            let left = Self::from_memory(memory, lookup_range.start..mid, height - 1, from, hasher);
            let right =
                Self::from_memory(memory, mid..lookup_range.end, height - 1, midpoint, hasher);
            NonLeaf {
                hash: hasher.compress(&left.hash(), &right.hash()),
                left: Arc::new(left),
                right: Arc::new(right),
            }
        }
    }

    pub fn tree_from_memory(
        memory_dimensions: MemoryDimensions,
        memory: &MemoryImage<F>,
        hasher: &impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        // Construct a Vec that includes the address space in the label calculation,
        // representing the entire memory tree.
        let mut memory_partition: Vec<(u64, [F; CHUNK])> = Vec::new();
        for ((address_space, pointer), value) in memory.items() {
            if pointer as usize / CHUNK >= (1 << memory_dimensions.address_height) {
                continue;
            }
            debug_assert!(pointer as usize / CHUNK < (1 << memory_dimensions.address_height));
            debug_assert!(address_space >= memory_dimensions.as_offset);
            debug_assert!(
                address_space - memory_dimensions.as_offset < (1 << memory_dimensions.as_height)
            );
            let label = (address_space, pointer / CHUNK as u32);
            let index = memory_dimensions.label_to_index(label);
            if memory_partition.is_empty() || memory_partition.last().unwrap().0 != index {
                memory_partition.push((index, [F::ZERO; CHUNK]));
            }
            let chunk = memory_partition.last_mut().unwrap();
            chunk.1[(pointer % CHUNK as u32) as usize] = value;
        }
        debug_assert!(memory_partition.is_sorted_by_key(|(addr, _)| addr));
        debug_assert!(
            memory_partition.last().map_or(0, |(addr, _)| *addr)
                < (1 << memory_dimensions.overall_height())
        );
        Self::from_memory(
            &memory_partition,
            0..memory_partition.len(),
            memory_dimensions.overall_height(),
            0,
            hasher,
        )
    }
}
