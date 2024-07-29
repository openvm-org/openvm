use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::ExpandChip;
use crate::memory::tree::{Hasher, MemoryNode};
use crate::memory::tree::MemoryNode::NonLeaf;

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn generate_trace_and_final_tree(
        &self,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
        hasher: &mut impl Hasher<CHUNK, F>,
    ) -> (RowMajorMatrix<F>, MemoryNode<CHUNK, F>) {
        let mut rows = vec![];
        let mut tree_helper = TreeHelper {
            address_height: self.air.address_height,
            as_offset: self.air.as_offset,
            final_memory,
            touched_nodes: &self.touched_nodes,
            trace_rows: &mut rows,
            precomputed_inverses: (0..self.air.address_height)
                .map(|h| {
                    F::from_canonical_usize(self.air.address_height - h)
                        .neg()
                        .inverse()
                })
                .collect(),
        };
        let final_tree = tree_helper.recur(
            self.air.as_height + self.air.address_height,
            self.initial_tree.clone(),
            0,
            0,
            hasher,
        );
        while rows.len() != trace_degree * ExpandCols::<CHUNK, F>::get_width() {
            rows.extend(unused_row::<CHUNK, F>().flatten());
        }
        let trace = RowMajorMatrix::new(rows, ExpandCols::<CHUNK, F>::get_width());
        (trace, final_tree)
    }
}

fn unused_row<const CHUNK: usize, F: PrimeField32>() -> ExpandCols<CHUNK, F> {
    let mut cols = ExpandCols::from_slice(&vec![F::zero(); ExpandCols::<CHUNK, F>::get_width()]);
    cols.children_height_section = F::one();
    cols
}

struct TreeHelper<'a, const CHUNK: usize, F: PrimeField32> {
    address_height: usize,
    as_offset: usize,
    final_memory: &'a HashMap<(F, F), F>,
    touched_nodes: &'a HashSet<(usize, usize, usize)>,
    trace_rows: &'a mut Vec<F>,
    // i -> 1 / (address_height - i)
    precomputed_inverses: Vec<F>,
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
            let address_space =
                F::from_canonical_usize((as_label >> self.address_height) + self.as_offset);
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

            let right_as_label = (2 * as_label) + if height > self.address_height { 1 } else { 0 };
            let right_address_label =
                (2 * address_label) + if height > self.address_height { 0 } else { 1 };
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
        let children_height = parent_height - 1;
        let children_height_section = if children_height >= self.address_height {
            1
        } else {
            0
        };
        let children_height_within =
            children_height - (self.address_height * children_height_section);
        let cols = if let NonLeaf { hash, left, right } = node {
            ExpandCols {
                expand_direction: if direction_changes.is_none() {
                    F::one()
                } else {
                    F::neg_one()
                },
                children_height_section: F::from_canonical_usize(children_height_section),
                children_height_within: F::from_canonical_usize(children_height_within),
                height_inverse: if children_height_section == 0 {
                    self.precomputed_inverses[children_height_within]
                } else {
                    F::zero()
                },
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
        self.trace_rows.extend(cols.flatten());
    }
}
