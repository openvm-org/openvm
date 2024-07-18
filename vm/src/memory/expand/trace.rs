use std::array::from_fn;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::ExpandChip;
use crate::memory::tree::MemoryNode::Leaf;
use crate::memory::tree::MemoryNode::NonLeaf;
use crate::memory::tree::{HashProvider, MemoryNode};

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn generate_trace_and_final_tree(
        &self,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
        hash_provider: &mut impl HashProvider<CHUNK, F>,
    ) -> (RowMajorMatrix<F>, HashMap<F, MemoryNode<CHUNK, F>>) {
        let mut rows = vec![];
        let mut final_trees = HashMap::new();
        for (address_space, initial_tree) in self.initial_trees.clone() {
            let mut tree_helper = TreeHelper {
                address_space,
                final_memory,
                touched_nodes: &self.touched_nodes,
                trace_rows: &mut rows,
            };
            final_trees.insert(
                address_space,
                tree_helper.recur(self.height, initial_tree, 0, hash_provider),
            );
        }
        while rows.len() != trace_degree * ExpandCols::<CHUNK, F>::get_width() {
            rows.extend(unused_row(hash_provider).flatten());
        }
        let trace = RowMajorMatrix::new(rows, ExpandCols::<CHUNK, F>::get_width());
        (trace, final_trees)
    }
}

fn unused_row<const CHUNK: usize, F: PrimeField32>(
    hash_provider: &mut impl HashProvider<CHUNK, F>,
) -> ExpandCols<CHUNK, F> {
    let mut result = ExpandCols::from_slice(&vec![F::zero(); ExpandCols::<CHUNK, F>::get_width()]);
    result.parent_hash = hash_provider.hash([F::zero(); CHUNK], [F::zero(); CHUNK]);
    result
}

struct TreeHelper<'a, const CHUNK: usize, F: PrimeField32> {
    address_space: F,
    final_memory: &'a HashMap<(F, F), F>,
    touched_nodes: &'a HashSet<(F, usize, usize)>,
    trace_rows: &'a mut Vec<F>,
}

impl<'a, const CHUNK: usize, F: PrimeField32> TreeHelper<'a, CHUNK, F> {
    fn recur(
        &mut self,
        height: usize,
        initial_node: MemoryNode<CHUNK, F>,
        label: usize,
        hash_provider: &mut impl HashProvider<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            Leaf(from_fn(|i| {
                *self
                    .final_memory
                    .get(&(
                        self.address_space,
                        F::from_canonical_usize(CHUNK * label) + F::from_canonical_usize(i),
                    ))
                    .unwrap_or(&F::zero())
            }))
        } else if let NonLeaf(_, initial_children) = initial_node.clone() {
            hash_provider.hash(initial_children[0].hash(), initial_children[1].hash());

            let labels = from_fn(|i| (2 * label) + i);
            let are_final = labels.map(|label| {
                !self
                    .touched_nodes
                    .contains(&(self.address_space, height - 1, label))
            });
            let final_children = from_fn(|i| {
                if are_final[i] {
                    initial_children[i].clone()
                } else {
                    Arc::new(self.recur(
                        height - 1,
                        (*initial_children[i]).clone(),
                        labels[i],
                        hash_provider,
                    ))
                }
            });

            let final_node = MemoryNode::new_nonleaf(final_children, hash_provider);
            self.add_trace_row(height, label, initial_node, Some(are_final));
            self.add_trace_row(height, label, final_node.clone(), None);
            final_node
        } else {
            panic!("Leaf {:?} found at nonzero height {}", initial_node, height);
        }
    }

    /// Expects `node` to be NonLeaf
    fn add_trace_row(
        &mut self,
        height: usize,
        label: usize,
        node: MemoryNode<CHUNK, F>,
        are_final: Option<[bool; 2]>,
    ) {
        let cols = if let NonLeaf(hash, children) = node {
            ExpandCols {
                direction: if are_final.is_some() {
                    F::one()
                } else {
                    F::neg_one()
                },
                address_space: self.address_space,
                parent_height: F::from_canonical_usize(height),
                parent_label: F::from_canonical_usize(label),
                parent_hash: hash,
                child_hashes: children.map(|child| child.hash()),
                are_final: are_final.unwrap_or([false; 2]).map(F::from_bool),
            }
        } else {
            panic!("trace_rows expects node = {:?} to be NonLeaf", node);
        };
        self.trace_rows.extend(cols.flatten());
    }
}
