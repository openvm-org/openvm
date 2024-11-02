use std::{borrow::BorrowMut, cmp::Reverse, collections::HashSet, sync::Arc};

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    arch::hasher::HasherChip,
    system::memory::{
        manager::dimensions::MemoryDimensions,
        merkle::{MemoryMerkleChip, MemoryMerkleCols},
        tree::MemoryNode::{self, NonLeaf},
        Equipartition,
    },
};

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    pub fn generate_trace_and_final_tree(
        &mut self,
        initial_tree: &MemoryNode<CHUNK, F>,
        final_memory: &Equipartition<F, CHUNK>,
        hasher: &mut impl HasherChip<CHUNK, F>,
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
            initial_tree,
            0,
            0,
            hasher,
        );
        // important that this sort be stable,
        // because we need the initial root to be first and the final root to be second
        rows.sort_by_key(|row| Reverse(row.parent_height));

        let width = MemoryMerkleCols::<F, CHUNK>::width();
        let height = rows.len().next_power_of_two();
        let mut trace = vec![F::zero(); width * height];

        for (trace_row, row) in trace.chunks_exact_mut(width).zip(rows) {
            *trace_row.borrow_mut() = row;
        }

        let trace = RowMajorMatrix::new(trace, width);
        (trace, final_tree)
    }
}

struct TreeHelper<'a, const CHUNK: usize, F: PrimeField32> {
    memory_dimensions: MemoryDimensions,
    final_memory: &'a Equipartition<F, CHUNK>,
    touched_nodes: &'a HashSet<(usize, usize, usize)>,
    trace_rows: &'a mut Vec<MemoryMerkleCols<F, CHUNK>>,
}

impl<const CHUNK: usize, F: PrimeField32> TreeHelper<'_, CHUNK, F> {
    fn recur(
        &mut self,
        height: usize,
        initial_node: &MemoryNode<CHUNK, F>,
        as_label: usize,
        address_label: usize,
        hasher: &mut impl HasherChip<CHUNK, F>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            let address_space =
                F::from_canonical_usize(as_label + self.memory_dimensions.as_offset);
            let leaf_values = *self
                .final_memory
                .get(&(address_space, address_label))
                .unwrap_or(&[F::zero(); CHUNK]);
            MemoryNode::new_leaf(hasher.hash(&leaf_values))
        } else if let NonLeaf {
            left: initial_left_node,
            right: initial_right_node,
            ..
        } = initial_node.clone()
        {
            // Tell the hasher about this hash.
            hasher.compress_and_record(&initial_left_node.hash(), &initial_right_node.hash());

            let is_as_section = height > self.memory_dimensions.address_height;

            let (left_as_label, right_as_label) = if is_as_section {
                (2 * as_label, 2 * as_label + 1)
            } else {
                (as_label, as_label)
            };
            let (left_address_label, right_address_label) = if is_as_section {
                (address_label, address_label)
            } else {
                (2 * address_label, 2 * address_label + 1)
            };

            let left_is_final =
                !self
                    .touched_nodes
                    .contains(&(height - 1, left_as_label, left_address_label));

            let final_left_node = if left_is_final {
                initial_left_node
            } else {
                Arc::new(self.recur(
                    height - 1,
                    &initial_left_node,
                    left_as_label,
                    left_address_label,
                    hasher,
                ))
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
                    &initial_right_node,
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
                &final_node,
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
        node: &MemoryNode<CHUNK, F>,
        direction_changes: Option<[bool; 2]>,
    ) {
        let [left_direction_change, right_direction_change] =
            direction_changes.unwrap_or([false; 2]);
        let cols = if let NonLeaf { hash, left, right } = node {
            MemoryMerkleCols {
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
                parent_hash: *hash,
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
