pub mod public_values;

use std::{collections::BTreeMap, sync::Arc};

use ax_stark_backend::p3_field::PrimeField32;
use MemoryNode::*;

use super::manager::dimensions::MemoryDimensions;
use crate::{
    arch::hasher::{Hasher, HasherChip},
    system::memory::Equipartition,
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
        memory: &BTreeMap<usize, [F; CHUNK]>,
        height: usize,
        from: usize,
        hasher: &impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        let mut range = memory.range(from..from + (1 << height));
        if height == 0 {
            let values = *memory.get(&from).unwrap_or(&[F::ZERO; CHUNK]);
            MemoryNode::new_leaf(hasher.hash(&values))
        } else if range.next().is_none() {
            let leaf_value = hasher.hash(&[F::ZERO; CHUNK]);
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
        hasher: &impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        // Construct a BTreeMap that includes the address space in the label calculation,
        // representing the entire memory tree.
        let mut memory_modified = BTreeMap::new();
        for (&label, &values) in memory {
            let index = memory_dimensions.label_to_index(label);
            memory_modified.insert(index, values);
        }
        Self::from_memory(
            &memory_modified,
            memory_dimensions.overall_height(),
            0,
            hasher,
        )
    }
}
