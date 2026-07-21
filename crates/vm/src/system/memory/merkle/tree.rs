use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
};
use rustc_hash::FxHashMap;

use super::{FinalState, MemoryMerkleCols};
use crate::{
    arch::hasher::{Hasher, HasherChip},
    system::memory::{
        dimensions::MemoryDimensions, merkle::memory_to_vec_partition, persistent::DirtyLeaves,
        AddressMap, Equipartition,
    },
};

fn parent_label_parts(
    md: &MemoryDimensions,
    tree_height: usize,
    parent_index: u64,
    parent_height: usize,
) -> (u32, u32) {
    let parent_depth = tree_height - parent_height;
    let heap_root_bit = 1u64 << parent_depth;
    debug_assert_ne!(parent_index & heap_root_bit, 0);
    let path = parent_index & !heap_root_bit;
    let address_bits_below = md.address_height.saturating_sub(parent_height);
    let address_mask = (1u64 << address_bits_below) - 1;
    let parent_address_label = (path & address_mask) as u32;
    let parent_as_label = (path >> address_bits_below) as u32;

    (parent_as_label, parent_address_label)
}

#[derive(Debug)]
pub struct MerkleTree<F, const DIGEST_WIDTH: usize> {
    /// Height of the tree -- the root is the only node at height `height`,
    /// and the leaves are at height `0`.
    height: usize,
    /// Nodes corresponding to all zeroes.
    zero_nodes: Vec<[F; DIGEST_WIDTH]>,
    /// Nodes in the tree that have ever been touched.
    nodes: FxHashMap<u64, [F; DIGEST_WIDTH]>,
}

impl<F: PrimeField32, const DIGEST_WIDTH: usize> MerkleTree<F, DIGEST_WIDTH> {
    pub fn new(height: usize, hasher: &impl Hasher<DIGEST_WIDTH, F>) -> Self {
        Self {
            height,
            zero_nodes: (0..height + 1)
                .scan(hasher.hash(&[F::ZERO; DIGEST_WIDTH]), |acc, _| {
                    let result = Some(*acc);
                    *acc = hasher.compress(acc, acc);
                    result
                })
                .collect(),
            nodes: FxHashMap::default(),
        }
    }

    pub fn root(&self) -> [F; DIGEST_WIDTH] {
        self.get_node(1)
    }

    pub fn get_node(&self, index: u64) -> [F; DIGEST_WIDTH] {
        self.nodes
            .get(&index)
            .copied()
            .unwrap_or(self.zero_nodes[self.height - index.ilog2() as usize])
    }

    fn get_node_at_height(&self, index: u64, node_height: usize) -> [F; DIGEST_WIDTH] {
        self.nodes
            .get(&index)
            .copied()
            .unwrap_or(self.zero_nodes[node_height])
    }

    #[allow(clippy::type_complexity)]
    /// Applies leaf updates upward to the root.
    ///
    /// Each layer entry carries an `is_dirty` bit: for leaves it originates at the
    /// boundary chip (value inequality), and for inner nodes it is the OR of the
    /// children's bits. A node emits a final-direction row (and records a final
    /// compression) only if it is dirty — or if it is the root, whose final row is
    /// pinned to the public values.
    fn process_layers<CompressFn>(
        &mut self,
        layer: Vec<(u64, [F; DIGEST_WIDTH], bool)>,
        md: &MemoryDimensions,
        mut rows: Option<&mut Vec<MemoryMerkleCols<F, DIGEST_WIDTH>>>,
        compress: CompressFn,
    ) where
        CompressFn: Fn(&[F; DIGEST_WIDTH], &[F; DIGEST_WIDTH]) -> [F; DIGEST_WIDTH] + Send + Sync,
    {
        debug_assert_eq!(self.height, md.overall_height());

        let initial_len = layer.len();
        let mut new_entries = layer;
        new_entries.reserve(initial_len.max(self.height));
        let mut layer = new_entries
            .par_iter()
            .map(|(index, values, is_dirty)| {
                let old_values = self.nodes.get(index).unwrap_or(&self.zero_nodes[0]);
                (*index, *values, *old_values, *is_dirty)
            })
            .collect::<Vec<_>>();
        for height in 1..=self.height {
            let new_layer = layer
                .iter()
                .enumerate()
                .filter_map(|(i, (index, values, old_values, is_dirty))| {
                    if i > 0 && layer[i - 1].0 ^ 1 == *index {
                        return None;
                    }

                    let par_index = index >> 1;

                    if i + 1 < layer.len() && layer[i + 1].0 == index ^ 1 {
                        let (_, sibling_values, sibling_old_values, sibling_dirty) = &layer[i + 1];
                        Some((
                            par_index,
                            Some((values, old_values, *is_dirty)),
                            Some((sibling_values, sibling_old_values, *sibling_dirty)),
                        ))
                    } else if index & 1 == 0 {
                        Some((par_index, Some((values, old_values, *is_dirty)), None))
                    } else {
                        Some((par_index, None, Some((values, old_values, *is_dirty))))
                    }
                })
                .collect::<Vec<_>>();

            match rows {
                None => {
                    layer = new_layer
                        .into_par_iter()
                        .map(|(par_index, left, right)| {
                            let left_node;
                            let (left, left_dirty) = if let Some((values, _, dirty)) = left {
                                (values, dirty)
                            } else {
                                left_node = self.get_node_at_height(2 * par_index, height - 1);
                                (&left_node, false)
                            };
                            let right_node;
                            let (right, right_dirty) = if let Some((values, _, dirty)) = right {
                                (values, dirty)
                            } else {
                                right_node = self.get_node_at_height(2 * par_index + 1, height - 1);
                                (&right_node, false)
                            };
                            let combined = compress(left, right);
                            let par_old_values = self.get_node_at_height(par_index, height);
                            (
                                par_index,
                                combined,
                                par_old_values,
                                left_dirty || right_dirty,
                            )
                        })
                        .collect();
                }
                Some(ref mut rows) => {
                    let height_section = F::from_bool(height > md.address_height);
                    let parent_height = F::from_usize(height);
                    let is_root = height == md.overall_height();
                    let (tmp, new_rows): (
                        Vec<(u64, [F; DIGEST_WIDTH], [F; DIGEST_WIDTH], bool)>,
                        Vec<_>,
                    ) = new_layer
                        .into_par_iter()
                        .map(|(par_index, left, right)| {
                            let (parent_as_label, parent_address_label) =
                                parent_label_parts(md, self.height, par_index, height);
                            // `changed_*` says the child is on the touched path (its
                            // layer entry exists); `*_dirty` says its value actually
                            // changed. Untouched children carry `dirty = false`.
                            let left_node;
                            let (left, old_left, left_dirty, changed_left) = match left {
                                Some((left, old_left, dirty)) => (left, old_left, dirty, true),
                                None => {
                                    left_node = self.get_node_at_height(2 * par_index, height - 1);
                                    (&left_node, &left_node, false, false)
                                }
                            };
                            let right_node;
                            let (right, old_right, right_dirty, changed_right) = match right {
                                Some((right, old_right, dirty)) => (right, old_right, dirty, true),
                                None => {
                                    right_node =
                                        self.get_node_at_height(2 * par_index + 1, height - 1);
                                    (&right_node, &right_node, false, false)
                                }
                            };
                            let node_dirty = left_dirty || right_dirty;
                            // Final rows are emitted only for dirty nodes, except the
                            // root: the AIR pins the first two rows to the initial/final
                            // root public values, so the root's final row always exists.
                            // A forced root row does not redefine a clean root as dirty:
                            // `node_dirty`, not `emits_final`, is what propagates upward.
                            let emits_final = node_dirty || is_root;

                            let par_old_values = self.get_node_at_height(par_index, height);
                            // Record the initial (old-children) compression for the
                            // initial row.
                            compress(old_left, old_right);
                            // Only emitted final rows claim a final compression. A clean
                            // node's new hash equals its stored one, so skip hashing.
                            let combined = if emits_final {
                                compress(left, right)
                            } else {
                                par_old_values
                            };

                            let initial_row = MemoryMerkleCols {
                                expand_direction: F::ONE,
                                height_section,
                                parent_height,
                                parent_height_inv: parent_height.inverse(),
                                is_root: F::from_bool(is_root),
                                parent_as_label: F::from_u32(parent_as_label),
                                parent_address_label: F::from_u32(parent_address_label),
                                parent_hash: par_old_values,
                                left_child_hash: *old_left,
                                right_child_hash: *old_right,
                                left_direction_different: F::ZERO,
                                right_direction_different: F::ZERO,
                                // Reference counts (see MemoryMerkleCols docs): a
                                // touched-clean child that our final row dd-borrows is
                                // consumed twice; an untouched child with no final row
                                // to prop it is not consumed.
                                left_extra_ref: F::from_bool(
                                    emits_final && changed_left && !left_dirty,
                                ),
                                right_extra_ref: F::from_bool(
                                    emits_final && changed_right && !right_dirty,
                                ),
                                left_absent_ref: F::from_bool(!emits_final && !changed_left),
                                right_absent_ref: F::from_bool(!emits_final && !changed_right),
                            };
                            let final_row = emits_final.then(|| MemoryMerkleCols {
                                expand_direction: F::NEG_ONE,
                                height_section,
                                parent_height,
                                parent_height_inv: parent_height.inverse(),
                                is_root: F::from_bool(is_root),
                                parent_as_label: F::from_u32(parent_as_label),
                                parent_address_label: F::from_u32(parent_address_label),
                                parent_hash: combined,
                                left_child_hash: *left,
                                right_child_hash: *right,
                                // dd = "not expanded finally": untouched *or*
                                // touched-clean children are borrowed from the initial
                                // tree.
                                left_direction_different: F::from_bool(!left_dirty),
                                right_direction_different: F::from_bool(!right_dirty),
                                left_extra_ref: F::ZERO,
                                right_extra_ref: F::ZERO,
                                left_absent_ref: F::ZERO,
                                right_absent_ref: F::ZERO,
                            });
                            (
                                (par_index, combined, par_old_values, node_dirty),
                                (initial_row, final_row),
                            )
                        })
                        .unzip();
                    rows.extend(
                        new_rows
                            .into_iter()
                            .flat_map(|(initial, fin)| std::iter::once(initial).chain(fin)),
                    );
                    layer = tmp;
                }
            }
            new_entries.extend(
                layer
                    .iter()
                    .map(|(idx, values, _, is_dirty)| (*idx, *values, *is_dirty)),
            );
        }

        self.nodes.reserve(new_entries.len());
        self.nodes.extend(
            new_entries
                .into_iter()
                .map(|(idx, values, _)| (idx, values)),
        );
    }

    pub fn from_memory(
        memory: &AddressMap,
        md: &MemoryDimensions,
        hasher: &(impl Hasher<DIGEST_WIDTH, F> + Sync),
    ) -> Self {
        let mut tree = Self::new(md.overall_height(), hasher);
        let layer: Vec<_> = memory_to_vec_partition(memory, md)
            .par_iter()
            .map(|(idx, v)| ((1 << tree.height) + idx, hasher.hash(v), false))
            .collect();
        tree.process_layers(layer, md, None, |left, right| hasher.compress(left, right));
        tree
    }

    /// `dirty_leaves` must be the set committed to by the boundary chip (keyed by
    /// `(address_space, leaf_ptr)`, like `touched`): its rows only reference a leaf's
    /// final state when that leaf is in the set.
    pub fn finalize(
        &mut self,
        hasher: &impl HasherChip<DIGEST_WIDTH, F>,
        touched: &Equipartition<F, DIGEST_WIDTH>,
        dirty_leaves: &DirtyLeaves,
        md: &MemoryDimensions,
    ) -> FinalState<DIGEST_WIDTH, F> {
        let init_root = self.get_node(1);
        let layer: Vec<_> = if !touched.is_empty() {
            touched
                .iter()
                .map(|((addr_sp, ptr), v)| {
                    (
                        (1 << self.height)
                            + md.label_to_index((*addr_sp, *ptr / DIGEST_WIDTH as u32)),
                        hasher.hash(v),
                        dirty_leaves.contains(&(*addr_sp, *ptr)),
                    )
                })
                .collect()
        } else {
            // Artificial touch to seed the walk so the root row pair exists. It is not
            // dirty, and no boundary row backs it (see the post-walk fix below).
            let index = 1 << self.height;
            vec![(index, self.get_node(index), false)]
        };
        // Upper bound: one initial row per touched-spanning node plus at most one final
        // row each.
        let touched_spanning_nodes = layer
            .iter()
            .zip(layer.iter().skip(1))
            .fold(md.overall_height(), |acc, ((lhs, _, _), (rhs, _, _))| {
                acc + (lhs ^ rhs).ilog2() as usize
            });
        let mut rows = Vec::with_capacity(2 * touched_spanning_nodes);
        self.process_layers(layer, md, Some(&mut rows), |left, right| {
            hasher.compress_and_record(left, right)
        });
        if touched.is_empty() {
            // The artificial touch seeds the walk so the root pair exists, but there is
            // no boundary row supplying the leaf's claim, so the height-1 initial row
            // (rows[0]) must treat the leaf as *untouched*: no extra reference if the
            // row's final counterpart exists (root case), an absent reference otherwise.
            rows[0].left_extra_ref = F::ZERO;
            rows[0].left_absent_ref = F::from_bool(rows[0].is_root == F::ZERO);
        }
        let final_root = self.get_node(1);
        FinalState {
            rows,
            init_root,
            final_root,
        }
    }

    pub fn top_tree(&self, top_height: usize) -> Vec<[F; DIGEST_WIDTH]> {
        // tree root is at index 1
        (0..(2 << top_height) - 1)
            .map(|i| self.get_node(i + 1))
            .collect()
    }
}
