use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::ExpandChip;
use crate::memory::tree::MemoryNode::NonLeaf;
use crate::memory::tree::{Hasher, MemoryNode};

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn generate_trace_and_final_tree(
        &self,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
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
                tree_helper.recur(self.height, initial_tree, 0, hasher),
            );
        }
        while rows.len() != trace_degree * ExpandCols::<CHUNK, F>::get_width() {
            rows.extend(unused_row(hasher).flatten());
        }
        let trace = RowMajorMatrix::new(rows, ExpandCols::<CHUNK, F>::get_width());
        (trace, final_trees)
    }
}

fn unused_row<const CHUNK: usize, F: PrimeField32>(
    hasher: &mut impl Hasher<CHUNK, F>,
) -> ExpandCols<CHUNK, F> {
    let mut result = ExpandCols::from_slice(&vec![F::zero(); ExpandCols::<CHUNK, F>::get_width()]);
    result.parent_hash = hasher.hash([F::zero(); CHUNK], [F::zero(); CHUNK]);
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
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            MemoryNode::new_leaf(std::array::from_fn(|i| {
                *self
                    .final_memory
                    .get(&(
                        self.address_space,
                        F::from_canonical_usize(CHUNK * label) + F::from_canonical_usize(i),
                    ))
                    .unwrap_or(&F::zero())
            }))
        } else if let NonLeaf {
            left: initial_left_node,
            right: initial_right_node,
            ..
        } = initial_node.clone()
        {
            hasher.hash(initial_left_node.hash(), initial_right_node.hash());

            let left_label = 2 * label;
            let left_is_final =
                !self
                    .touched_nodes
                    .contains(&(self.address_space, height - 1, left_label));
            let final_left_node = if left_is_final {
                initial_left_node
            } else {
                Arc::new(self.recur(height - 1, (*initial_left_node).clone(), left_label, hasher))
            };

            let right_label = (2 * label) + 1;
            let right_is_final =
                !self
                    .touched_nodes
                    .contains(&(self.address_space, height - 1, right_label));
            let final_right_node = if right_is_final {
                initial_right_node
            } else {
                Arc::new(self.recur(
                    height - 1,
                    (*initial_right_node).clone(),
                    right_label,
                    hasher,
                ))
            };

            let final_node = MemoryNode::new_nonleaf(final_left_node, final_right_node, hasher);
            self.add_trace_row(
                height,
                label,
                initial_node,
                Some([left_is_final, right_is_final]),
            );
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
        let [left_is_final, right_is_final] = are_final.unwrap_or([false; 2]);
        let cols = if let NonLeaf { hash, left, right } = node {
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
                left_child_hash: left.hash(),
                right_child_hash: right.hash(),
                left_is_final: F::from_bool(left_is_final),
                right_is_final: F::from_bool(right_is_final),
            }
        } else {
            panic!("trace_rows expects node = {:?} to be NonLeaf", node);
        };
        self.trace_rows.extend(cols.flatten());
    }
}
