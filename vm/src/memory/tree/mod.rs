use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use p3_field::PrimeField32;

use afs_chips::sub_chip::LocalTraceInstructions;
use MemoryNode::*;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};

fn hash<const CHUNK: usize, F: PrimeField32>(left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
    assert_eq!(CHUNK, 8);
    let air =
        Poseidon2Air::<16, F>::from_config(Poseidon2Config::<16, F>::new_p3_baby_bear_16(), 0);
    let input_state = [left, right].concat().try_into().unwrap();
    let internal = air.generate_trace_row(input_state);
    let output = internal.io.output.to_vec();
    output.try_into().unwrap()
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
    ) -> MemoryNode<CHUNK, F> {
        NonLeaf(hash(left.hash(), right.hash()), left, right)
    }

    pub fn construct_all_zeros(height: usize) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Leaf([F::zero(); CHUNK])
        } else {
            let child = Arc::new(Self::construct_all_zeros(height - 1));
            Self::new_nonleaf(child.clone(), child)
        }
    }

    pub fn from_memory(height: usize, memory: HashMap<usize, F>) -> MemoryNode<CHUNK, F> {
        if memory.is_empty() {
            Self::construct_all_zeros(height)
        } else if height == 0 {
            let mut values = [F::zero(); CHUNK];
            for (address, value) in memory {
                values[address] = value;
            }
            Leaf(values)
        } else {
            let midpoint: usize = 1 << (height - 1);
            let mut left_memory = HashMap::new();
            let mut right_memory = HashMap::new();
            for (address, value) in memory {
                if address < midpoint {
                    left_memory.insert(address, value);
                } else {
                    right_memory.insert(address - midpoint, value);
                }
            }
            let left = Self::from_memory(height - 1, left_memory);
            let right = Self::from_memory(height - 1, right_memory);
            Self::new_nonleaf(Arc::new(left), Arc::new(right))
        }
    }
}

pub fn trees_from_full_memory<const CHUNK: usize, F: PrimeField32>(
    height: usize,
    memory: &HashMap<(F, F), F>,
) -> HashMap<F, MemoryNode<CHUNK, F>> {
    let mut trees = HashMap::new();
    for &address_space in memory
        .keys()
        .map(|(address_space, _)| address_space)
        .collect::<HashSet<_>>()
    {
        let mut memory_here = HashMap::new();
        for (&(relevant_address_space, address), &value) in memory {
            if relevant_address_space == address_space {
                memory_here.insert(address.as_canonical_u64() as usize, value);
            }
        }
        trees.insert(address_space, MemoryNode::from_memory(height, memory_here));
    }
    trees
}
