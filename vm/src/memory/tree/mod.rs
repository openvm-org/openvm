use std::collections::HashMap;
use std::sync::Arc;

use p3_field::PrimeField32;

use MemoryNode::*;

pub trait HashProvider<const CHUNK: usize, F> {
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK];
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryNode<const CHUNK: usize, F: PrimeField32> {
    Leaf([F; CHUNK]),
    NonLeaf(
        [F; CHUNK],
        Arc<MemoryNode<CHUNK, F>>,
        Arc<MemoryNode<CHUNK, F>>,
    ),
}

impl<const CHUNK: usize, F: PrimeField32> MemoryNode<CHUNK, F> {
    pub fn hash(&self) -> [F; CHUNK] {
        match self {
            Leaf(hash) => *hash,
            NonLeaf(hash, ..) => *hash,
        }
    }

    pub fn new_nonleaf(
        left: Arc<MemoryNode<CHUNK, F>>,
        right: Arc<MemoryNode<CHUNK, F>>,
        hash_provider: &mut impl HashProvider<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        NonLeaf(hash_provider.hash(left.hash(), right.hash()), left, right)
    }

    pub fn construct_all_zeros(
        height: usize,
        hash_provider: &mut impl HashProvider<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Leaf([F::zero(); CHUNK])
        } else {
            let child = Arc::new(Self::construct_all_zeros(height - 1, hash_provider));
            Self::new_nonleaf(child.clone(), child, hash_provider)
        }
    }

    pub fn from_memory(
        height: usize,
        memory: HashMap<usize, F>,
        hash_provider: &mut impl HashProvider<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if memory.is_empty() {
            Self::construct_all_zeros(height, hash_provider)
        } else if height == 0 {
            let mut values = [F::zero(); CHUNK];
            for (address, value) in memory {
                values[address] = value;
            }
            Leaf(values)
        } else {
            let midpoint: usize = CHUNK << (height - 1);
            let mut left_memory = HashMap::new();
            let mut right_memory = HashMap::new();
            for (address, value) in memory {
                if address < midpoint {
                    left_memory.insert(address, value);
                } else {
                    right_memory.insert(address - midpoint, value);
                }
            }
            let left = Self::from_memory(height - 1, left_memory, hash_provider);
            let right = Self::from_memory(height - 1, right_memory, hash_provider);
            Self::new_nonleaf(Arc::new(left), Arc::new(right), hash_provider)
        }
    }
}

pub fn trees_from_full_memory<const CHUNK: usize, F: PrimeField32>(
    height: usize,
    address_spaces: &[F],
    memory: &HashMap<(F, F), F>,
    hash_provider: &mut impl HashProvider<CHUNK, F>,
) -> HashMap<F, MemoryNode<CHUNK, F>> {
    let mut trees = HashMap::new();
    for &address_space in address_spaces {
        let mut memory_here = HashMap::new();
        for (&(relevant_address_space, address), &value) in memory {
            if relevant_address_space == address_space {
                memory_here.insert(address.as_canonical_u64() as usize, value);
            }
        }
        trees.insert(
            address_space,
            MemoryNode::from_memory(height, memory_here, hash_provider),
        );
    }
    trees
}
