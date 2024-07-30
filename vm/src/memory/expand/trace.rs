use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::{ExpandChip, MemoryDimensions};
use crate::memory::tree::MemoryNode::NonLeaf;
use crate::memory::tree::{Hasher, MemoryNode};

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn generate_trace_and_final_tree(
        &mut self,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> (RowMajorMatrix<F>, MemoryNode<CHUNK, F>) {
        // there needs to be a touched node with `height_section` = 0
        // shouldn't be a leaf because
        // trace generation will expect an interaction from MemoryInterfaceChip in that case
        if self.touched_nodes.len() == 1 {
            self.touch_node(1, 0, 0);
        }

        let mut rows = vec![];
        let mut tree_helper = TreeHelper {
            memory_dimensions: self.air.memory_dimensions,
            final_memory,
            touched_nodes: &self.touched_nodes,
            trace_rows: &mut rows,
        };
        let final_tree = tree_helper.recur(
            self.air.memory_dimensions.overall_height(),
            self.initial_tree.clone(),
            0,
            0,
            hasher,
        );
        assert!(rows.len() <= trace_degree);
        while rows.len() != trace_degree {
            rows.push(unused_row::<CHUNK, F>());
        }
        // important that this sort be stable
        rows.sort_by_key(|row| -(row.parent_height.as_canonical_u64() as i32));
        let trace = RowMajorMatrix::new(
            rows.iter().flat_map(ExpandCols::flatten).collect(),
            ExpandCols::<CHUNK, F>::get_width(),
        );
        (trace, final_tree)
    }
}

fn unused_row<const CHUNK: usize, F: PrimeField32>() -> ExpandCols<CHUNK, F> {
    ExpandCols::from_slice(&vec![F::zero(); ExpandCols::<CHUNK, F>::get_width()])
}

struct TreeHelper<'a, const CHUNK: usize, F: PrimeField32> {
    memory_dimensions: MemoryDimensions,
    final_memory: &'a HashMap<(F, F), F>,
    touched_nodes: &'a HashSet<(usize, usize, usize)>,
    trace_rows: &'a mut Vec<ExpandCols<CHUNK, F>>,
}

impl<'a, const CHUNK: usize, F: PrimeField32> TreeHelper<'a, CHUNK, F> {
    fn recur(
        &mut self,
        height: usize,
        initial_node: MemoryNode<CHUNK, F>,
        as_label: usize,
        address_label: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            let address_space = F::from_canonical_usize(
                (as_label >> self.memory_dimensions.address_height)
                    + self.memory_dimensions.as_offset,
            );
            MemoryNode::new_leaf(std::array::from_fn(|i| {
                *self
                    .final_memory
                    .get(&(
                        address_space,
                        F::from_canonical_usize((CHUNK * address_label) + i),
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

            let left_as_label = 2 * as_label;
            let left_address_label = 2 * address_label;
            let left_is_final =
                !self
                    .touched_nodes
                    .contains(&(height - 1, left_as_label, left_address_label));
            let final_left_node = if left_is_final {
                initial_left_node
            } else {
                Arc::new(self.recur(
                    height - 1,
                    (*initial_left_node).clone(),
                    left_as_label,
                    left_address_label,
                    hasher,
                ))
            };

            let right_as_label = (2 * as_label)
                + if height > self.memory_dimensions.address_height {
                    1
                } else {
                    0
                };
            let right_address_label = (2 * address_label)
                + if height > self.memory_dimensions.address_height {
                    0
                } else {
                    1
                };
            let right_is_final =
                !self
                    .touched_nodes
                    .contains(&(height - 1, right_as_label, right_address_label));
            let final_right_node = if right_is_final {
                initial_right_node
            } else {
                Arc::new(self.recur(
                    height - 1,
                    (*initial_right_node).clone(),
                    right_as_label,
                    right_address_label,
                    hasher,
                ))
            };

            let final_node = MemoryNode::new_nonleaf(final_left_node, final_right_node, hasher);
            self.add_trace_row(height, as_label, address_label, initial_node, None);
            self.add_trace_row(
                height,
                as_label,
                address_label,
                final_node.clone(),
                Some([left_is_final, right_is_final]),
            );
            final_node
        } else {
            panic!("Leaf {:?} found at nonzero height {}", initial_node, height);
        }
    }

    /// Expects `node` to be NonLeaf
    fn add_trace_row(
        &mut self,
        parent_height: usize,
        as_label: usize,
        address_label: usize,
        node: MemoryNode<CHUNK, F>,
        direction_changes: Option<[bool; 2]>,
    ) {
        let [left_direction_change, right_direction_change] =
            direction_changes.unwrap_or([false; 2]);
        let cols = if let NonLeaf { hash, left, right } = node {
            ExpandCols {
                expand_direction: if direction_changes.is_none() {
                    F::one()
                } else {
                    F::neg_one()
                },
                height_section: F::from_bool(parent_height > self.memory_dimensions.address_height),
                parent_height: F::from_canonical_usize(parent_height),
                is_root: F::from_bool(parent_height == self.memory_dimensions.overall_height()),
                parent_as_label: F::from_canonical_usize(as_label),
                parent_address_label: F::from_canonical_usize(address_label),
                parent_hash: hash,
                left_child_hash: left.hash(),
                right_child_hash: right.hash(),
                left_direction_different: F::from_bool(left_direction_change),
                right_direction_different: F::from_bool(right_direction_change),
            }
        } else {
            panic!("trace_rows expects node = {:?} to be NonLeaf", node);
        };
        self.trace_rows.push(cols);
    }
}
