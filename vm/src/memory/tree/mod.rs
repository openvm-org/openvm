use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use p3_field::PrimeField32;
use MemoryNode::*;

use super::manager::dimensions::MemoryDimensions;

pub trait Hasher<const CHUNK: usize, F> {
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK];
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
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> Self {
        NonLeaf {
            hash: hasher.hash(left.hash(), right.hash()),
            left,
            right,
        }
    }

    pub fn construct_all_zeros(
        height: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Self::new_leaf([F::zero(); CHUNK])
        } else {
            let child = Arc::new(Self::construct_all_zeros(height - 1, hasher));
            Self::new_nonleaf(child.clone(), child, hasher)
        }
    }

    fn from_memory(
        memory: &BTreeMap<usize, F>,
        height: usize,
        from: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        let mut range = memory.range(from..from + (CHUNK << height));
        if height == 0 {
            let mut values = [F::zero(); CHUNK];
            for (&address, &value) in range {
                values[address - from] = value;
            }
            MemoryNode::new_leaf(values)
        } else if range.next().is_none() {
            MemoryNode::construct_all_zeros(height, hasher)
        } else {
            let midpoint = from + (CHUNK << (height - 1));
            let left = Self::from_memory(memory, height - 1, from, hasher);
            let right = Self::from_memory(memory, height - 1, midpoint, hasher);
            MemoryNode::new_nonleaf(Arc::new(left), Arc::new(right), hasher)
        }
    }

    pub fn tree_from_memory(
        memory_dimensions: MemoryDimensions,
        memory: &HashMap<(F, F), F>,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        let mut memory_modified = BTreeMap::new();
        for (&(address_space, address), &value) in memory {
            let complete_address = (((address_space.as_canonical_u64() as usize)
                - memory_dimensions.as_offset)
                * (CHUNK << memory_dimensions.address_height))
                + (address.as_canonical_u64() as usize);
            memory_modified.insert(complete_address, value);
        }
        Self::from_memory(
            &memory_modified,
            memory_dimensions.overall_height(),
            0,
            hasher,
        )
    }
}
