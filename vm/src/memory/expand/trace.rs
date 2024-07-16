use std::array::from_fn;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use p3_field::{Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::ExpandAir;
use crate::memory::tree::MemoryNode;
use crate::memory::tree::MemoryNode::Leaf;
use crate::memory::tree::MemoryNode::NonLeaf;

impl<const CHUNK: usize> ExpandAir<CHUNK> {
    pub fn generate_trace_and_final_tree<F: PrimeField64>(
        &mut self,
        initial_trees: HashMap<F, MemoryNode<CHUNK, F>>,
        touched_addresses: Vec<(F, F)>,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
    ) -> (RowMajorMatrix<F>, HashMap<F, MemoryNode<CHUNK, F>>) {
        let mut touched_nodes_map = HashMap::new();
        for address_space in initial_trees.keys() {
            touched_nodes_map.insert(address_space, HashSet::new());
        }
        for (address_space, address) in touched_addresses {
            let mut node_label = address.as_canonical_u64() as usize;
            let touched_nodes = touched_nodes_map.get_mut(&address_space).unwrap();
            for _ in 0..self.height {
                if !touched_nodes.insert(node_label) {
                    // this break only necessary for optimization
                    break;
                }
                node_label /= 2;
            }
        }
        let mut rows = vec![];
        let mut final_trees = HashMap::new();
        for &address_space in initial_trees.keys() {
            let touched_nodes = &touched_nodes_map[&address_space];
            final_trees.insert(
                address_space,
                recur(
                    self.height,
                    initial_trees[&address_space].clone(),
                    0,
                    address_space,
                    final_memory,
                    touched_nodes,
                    &mut rows,
                ),
            );
        }
        let trace = RowMajorMatrix::new(rows, ExpandCols::<CHUNK, F>::get_width());
        (trace, final_trees)
    }
}

/// Expects `initial_node`, `final_node` to be NonLeaf
fn add_trace_rows<const CHUNK: usize, F: Field>(
    height: usize,
    label: usize,
    initial_node: MemoryNode<CHUNK, F>,
    final_node: MemoryNode<CHUNK, F>,
    left_is_final: bool,
    right_is_final: bool,
    address_space: F,
    trace_rows: &mut Vec<F>,
) {
    let initial_cols = if let NonLeaf(hash, left, right) = initial_node {
        ExpandCols {
            multiplicity: F::one(),
            is_compress: F::neg_one(),
            address_space,
            parent_height: F::from_canonical_usize(height - 1),
            parent_label: F::from_canonical_usize(label),
            parent_hash: hash,
            left_child_hash: left.hash(),
            right_child_hash: right.hash(),
            left_is_final: F::from_bool(left_is_final),
            right_is_final: F::from_bool(right_is_final),
        }
    } else {
        panic!(
            "trace_rows expects initial_node = {:?} to be NonLeaf",
            initial_node
        );
    };
    let final_cols = if let NonLeaf(hash, left, right) = final_node {
        ExpandCols {
            multiplicity: F::neg_one(),
            is_compress: F::one(),
            address_space,
            parent_height: F::from_canonical_usize(height - 1),
            parent_label: F::from_canonical_usize(label),
            parent_hash: hash,
            left_child_hash: left.hash(),
            right_child_hash: right.hash(),
            left_is_final: F::zero(),
            right_is_final: F::zero(),
        }
    } else {
        panic!(
            "trace_rows expects final_node = {:?} to be NonLeaf",
            final_node
        );
    };
    trace_rows.extend(initial_cols.flatten());
    trace_rows.extend(final_cols.flatten());
}

fn recur<const CHUNK: usize, F: Field>(
    height: usize,
    initial_node: MemoryNode<CHUNK, F>,
    label: usize,
    address_space: F,
    final_memory: &HashMap<(F, F), F>,
    touched_nodes: &HashSet<usize>,
    trace_rows: &mut Vec<F>,
) -> MemoryNode<CHUNK, F> {
    if height == 0 {
        Leaf(from_fn(|i| {
            final_memory[&(
                address_space,
                F::from_canonical_usize(CHUNK * label) + F::from_canonical_usize(i),
            )]
        }))
    } else if let NonLeaf(_, initial_left_node, initial_right_node) = initial_node.clone() {
        let left_label = 2 * label;
        let left_is_final = !touched_nodes.contains(&left_label);
        let final_left_node = if left_is_final {
            initial_left_node
        } else {
            Arc::new(recur(
                height - 1,
                (*initial_left_node).clone(),
                left_label,
                address_space,
                final_memory,
                touched_nodes,
                trace_rows,
            ))
        };

        let right_label = (2 * label) + 1;
        let right_is_final = !touched_nodes.contains(&right_label);
        let final_right_node = if right_is_final {
            initial_right_node
        } else {
            Arc::new(recur(
                height - 1,
                (*initial_right_node).clone(),
                left_label,
                address_space,
                final_memory,
                touched_nodes,
                trace_rows,
            ))
        };

        let final_node = MemoryNode::new_nonleaf(final_left_node, final_right_node);
        add_trace_rows(
            height,
            label,
            initial_node,
            final_node.clone(),
            left_is_final,
            right_is_final,
            address_space,
            trace_rows,
        );
        final_node
    } else {
        panic!("Leaf {:?} found at nonzero height {}", initial_node, height);
    }
}
