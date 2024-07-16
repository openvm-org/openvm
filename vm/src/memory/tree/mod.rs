use std::sync::Arc;

use p3_field::Field;

use MemoryNode::*;

fn hash<const CHUNK: usize, F: Field>(left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
    todo!();
}

#[derive(Clone, Debug)]
pub enum MemoryNode<const CHUNK: usize, F: Field> {
    Leaf([F; CHUNK]),
    NonLeaf(
        [F; CHUNK],
        Arc<MemoryNode<CHUNK, F>>,
        Arc<MemoryNode<CHUNK, F>>,
    ),
}

impl<const CHUNK: usize, F: Field> MemoryNode<CHUNK, F> {
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
}
