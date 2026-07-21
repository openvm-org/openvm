use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
};
use rustc_hash::FxHashMap;

use super::{FinalState, MemoryMerkleCols};
use crate::{
    arch::hasher::{Hasher, HasherChip},
    system::memory::{
        dimensions::MemoryDimensions, merkle::memory_to_vec_partition, AddressMap, Equipartition,
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
    fn process_layers<CompressFn>(
        &mut self,
        layer: Vec<(u64, [F; DIGEST_WIDTH])>,
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
            .map(|(index, values)| {
                let old_values = self.nodes.get(index).unwrap_or(&self.zero_nodes[0]);
                (*index, *values, *old_values)
            })
            .collect::<Vec<_>>();
        for height in 1..=self.height {
            let new_layer = layer
                .iter()
                .enumerate()
                .filter_map(|(i, (index, values, old_values))| {
                    if i > 0 && layer[i - 1].0 ^ 1 == *index {
                        return None;
                    }

                    let par_index = index >> 1;

                    if i + 1 < layer.len() && layer[i + 1].0 == index ^ 1 {
                        let (_, sibling_values, sibling_old_values) = &layer[i + 1];
                        Some((
                            par_index,
                            Some((values, old_values)),
                            Some((sibling_values, sibling_old_values)),
                        ))
                    } else if index & 1 == 0 {
                        Some((par_index, Some((values, old_values)), None))
                    } else {
                        Some((par_index, None, Some((values, old_values))))
                    }
                })
                .collect::<Vec<_>>();

            match rows {
                None => {
                    layer = new_layer
                        .into_par_iter()
                        .map(|(par_index, left, right)| {
                            let left_node;
                            let left = if let Some(left) = left {
                                left.0
                            } else {
                                left_node = self.get_node_at_height(2 * par_index, height - 1);
                                &left_node
                            };
                            let right_node;
                            let right = if let Some(right) = right {
                                right.0
                            } else {
                                right_node = self.get_node_at_height(2 * par_index + 1, height - 1);
                                &right_node
                            };
                            let combined = compress(left, right);
                            let par_old_values = self.get_node_at_height(par_index, height);
                            (par_index, combined, par_old_values)
                        })
                        .collect();
                }
                Some(ref mut rows) => {
                    let height_section = F::from_bool(height > md.address_height);
                    let parent_height = F::from_usize(height);
                    let is_root = F::from_bool(height == md.overall_height());
                    let (tmp, new_rows): (
                        Vec<(u64, [F; DIGEST_WIDTH], [F; DIGEST_WIDTH])>,
                        Vec<[_; 2]>,
                    ) = new_layer
                        .into_par_iter()
                        .map(|(par_index, left, right)| {
                            let (parent_as_label, parent_address_label) =
                                parent_label_parts(md, self.height, par_index, height);
                            let left_node;
                            let (left, old_left, changed_left) = match left {
                                Some((left, old_left)) => (left, old_left, true),
                                None => {
                                    left_node = self.get_node_at_height(2 * par_index, height - 1);
                                    (&left_node, &left_node, false)
                                }
                            };
                            let right_node;
                            let (right, old_right, changed_right) = match right {
                                Some((right, old_right)) => (right, old_right, true),
                                None => {
                                    right_node =
                                        self.get_node_at_height(2 * par_index + 1, height - 1);
                                    (&right_node, &right_node, false)
                                }
                            };
                            let combined = compress(left, right);
                            // Record the old child pair on the compression bus.
                            compress(old_left, old_right);
                            let par_old_values = self.get_node_at_height(par_index, height);
                            (
                                (par_index, combined, par_old_values),
                                [
                                    MemoryMerkleCols {
                                        expand_direction: F::ONE,
                                        height_section,
                                        parent_height,
                                        parent_height_inv: parent_height.inverse(),
                                        is_root,
                                        parent_as_label: F::from_u32(parent_as_label),
                                        parent_address_label: F::from_u32(parent_address_label),
                                        parent_hash: par_old_values,
                                        left_child_hash: *old_left,
                                        right_child_hash: *old_right,
                                        left_direction_different: F::ZERO,
                                        right_direction_different: F::ZERO,
                                        left_extra_ref: F::ZERO,
                                        right_extra_ref: F::ZERO,
                                        left_absent_ref: F::ZERO,
                                        right_absent_ref: F::ZERO,
                                    },
                                    MemoryMerkleCols {
                                        expand_direction: F::NEG_ONE,
                                        height_section,
                                        parent_height,
                                        parent_height_inv: parent_height.inverse(),
                                        is_root,
                                        parent_as_label: F::from_u32(parent_as_label),
                                        parent_address_label: F::from_u32(parent_address_label),
                                        parent_hash: combined,
                                        left_child_hash: *left,
                                        right_child_hash: *right,
                                        left_direction_different: F::from_bool(!changed_left),
                                        right_direction_different: F::from_bool(!changed_right),
                                        left_extra_ref: F::ZERO,
                                        right_extra_ref: F::ZERO,
                                        left_absent_ref: F::ZERO,
                                        right_absent_ref: F::ZERO,
                                    },
                                ],
                            )
                        })
                        .unzip();
                    rows.extend(new_rows.into_iter().flatten());
                    layer = tmp;
                }
            }
            new_entries.extend(layer.iter().map(|(idx, values, _)| (*idx, *values)));
        }

        self.nodes.reserve(new_entries.len());
        self.nodes.extend(new_entries);
    }

    pub fn from_memory(
        memory: &AddressMap,
        md: &MemoryDimensions,
        hasher: &(impl Hasher<DIGEST_WIDTH, F> + Sync),
    ) -> Self {
        let mut tree = Self::new(md.overall_height(), hasher);
        let layer: Vec<_> = memory_to_vec_partition(memory, md)
            .par_iter()
            .map(|(idx, v)| ((1 << tree.height) + idx, hasher.hash(v)))
            .collect();
        tree.process_layers(layer, md, None, |left, right| hasher.compress(left, right));
        tree
    }

    pub fn finalize(
        &mut self,
        hasher: &impl HasherChip<DIGEST_WIDTH, F>,
        touched: &Equipartition<F, DIGEST_WIDTH>,
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
                    )
                })
                .collect()
        } else {
            let index = 1 << self.height;
            vec![(index, self.get_node(index))]
        };
        let mut rows = Vec::with_capacity(if layer.is_empty() {
            0
        } else {
            layer
                .iter()
                .zip(layer.iter().skip(1))
                .fold(md.overall_height(), |acc, ((lhs, _), (rhs, _))| {
                    acc + (lhs ^ rhs).ilog2() as usize
                })
        });
        self.process_layers(layer, md, Some(&mut rows), |left, right| {
            hasher.compress_and_record(left, right)
        });
        if touched.is_empty() {
            // If we made an artificial touch, we need to change the direction changes for the
            // leaves
            rows[1].left_direction_different = F::ONE;
            rows[1].right_direction_different = F::ONE;
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
