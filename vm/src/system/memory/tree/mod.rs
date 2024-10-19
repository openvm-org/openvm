use std::{collections::BTreeMap, sync::Arc};

use p3_field::{Field, PrimeField32};
use MemoryNode::*;

use super::manager::dimensions::MemoryDimensions;
use crate::system::memory::Equipartition;

pub trait HasherChip<const CHUNK: usize, F: Field> {
    /// Statelessly compresses two chunks of data into a single chunk.
    fn compress(&self, left: &[F; CHUNK], right: &[F; CHUNK]) -> [F; CHUNK];

    /// Stateful version of `hash` for recording the event in the chip.
    fn compress_and_record(&mut self, left: &[F; CHUNK], right: &[F; CHUNK]) -> [F; CHUNK];

    fn hash(&self, values: &[F; CHUNK]) -> [F; CHUNK] {
        self.compress(values, &[F::zero(); CHUNK])
    }

    fn hash_and_record(&mut self, values: &[F; CHUNK]) -> [F; CHUNK] {
        self.compress_and_record(values, &[F::zero(); CHUNK])
    }
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
            hash: hasher.compress_and_record(&left.hash(), &right.hash()),
            left,
            right,
        }
    }

    /// Returns a tree of height `height` with all leaves set to `leaf_value`.
    pub fn construct_uniform(
        height: usize,
        leaf_value: [F; CHUNK],
        hasher: &impl HasherChip<CHUNK, F>,
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
        memory: &BTreeMap<usize, [F; CHUNK]>,
        height: usize,
        from: usize,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        let mut range = memory.range(from..from + (1 << height));
        if height == 0 {
            let values = *memory.get(&from).unwrap_or(&[F::zero(); CHUNK]);
            MemoryNode::new_leaf(hasher.hash(&values))
        } else if range.next().is_none() {
            let leaf_value = hasher.hash(&[F::zero(); CHUNK]);
            MemoryNode::construct_uniform(height, leaf_value, hasher)
        } else {
            let midpoint = from + (1 << (height - 1));
            let left = Self::from_memory(memory, height - 1, from, hasher);
            let right = Self::from_memory(memory, height - 1, midpoint, hasher);
            NonLeaf {
                hash: hasher.compress(&left.hash(), &right.hash()),
                left: Arc::new(left),
                right: Arc::new(right),
            }
        }
    }

    pub fn tree_from_memory(
        memory_dimensions: MemoryDimensions,
        memory: &Equipartition<F, CHUNK>,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        // Construct a BTreeMap that includes the address space in the label calculation,
        // representing the entire memory tree.
        let mut memory_modified = BTreeMap::new();
        for (&(address_space, address_label), &values) in memory {
            let label = (((address_space.as_canonical_u32() as usize)
                - memory_dimensions.as_offset)
                << memory_dimensions.address_height)
                + address_label;
            memory_modified.insert(label, values);
        }
        Self::from_memory(
            &memory_modified,
            memory_dimensions.overall_height(),
            0,
            hasher,
        )
    }
}
