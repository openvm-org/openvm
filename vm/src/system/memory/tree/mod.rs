use std::{collections::BTreeMap, sync::Arc};

use p3_field::PrimeField32;
use MemoryNode::*;

use super::manager::dimensions::MemoryDimensions;
use crate::system::memory::Equipartition;

pub trait HasherChip<const CHUNK: usize, F> {
    /// Statelessly compresses two chunks of data into a single chunk.
    fn hash(&self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK];

    /// Stateful version of `hash` for recording the event in the chip.
    fn hash_and_record(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK];
}

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
            hash: hasher.hash_and_record(left.hash(), right.hash()),
            left,
            right,
        }
    }

    pub fn construct_all_zeros(
        height: usize,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Self::new_leaf([F::zero(); CHUNK])
        } else {
            let child = Arc::new(Self::construct_all_zeros(height - 1, hasher));
            NonLeaf {
                hash: hasher.hash(child.hash(), child.hash()),
                left: child.clone(),
                right: child,
            }
        }
    }

    fn from_memory(
        memory: &BTreeMap<u64, [F; CHUNK]>,
        height: usize,
        from: u64,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        let chunk_label = from >> 1;
        if height == 0 {
            // if from is even, we are at a leaf memory node; otherwise we are at a padding node
            if from & 1 == 0 {
                let values = *memory.get(&chunk_label).unwrap_or(&[F::zero(); CHUNK]);
                MemoryNode::new_leaf(values)
            } else {
                MemoryNode::new_leaf([F::zero(); CHUNK])
            }
        } else if memory
            .range(chunk_label..chunk_label + (1 << (height - 1)))
            .next()
            .is_none()
        {
            MemoryNode::construct_all_zeros(height, hasher)
        } else {
            let midpoint = from + (1 << (height - 1));
            let left = Self::from_memory(memory, height - 1, from, hasher);
            let right = Self::from_memory(memory, height - 1, midpoint, hasher);
            NonLeaf {
                hash: hasher.hash(left.hash(), right.hash()),
                left: Arc::new(left),
                right: Arc::new(right),
            }
        }
    }

    pub fn tree_from_memory(
        dims: MemoryDimensions,
        memory: &Equipartition<F, CHUNK>,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        // Construct a BTreeMap that includes the address space in the chunk label calculation,
        // representing the entire memory tree.
        let mut memory_modified = BTreeMap::new();
        for (&(address_space, chunk_label), &values) in memory {
            let first_chunk_label =
                (address_space.as_canonical_u64() - dims.as_offset as u64) << dims.address_height;
            memory_modified.insert(first_chunk_label + chunk_label as u64, values);
        }
        Self::from_memory(&memory_modified, dims.overall_height(), 0, hasher)
    }
}
