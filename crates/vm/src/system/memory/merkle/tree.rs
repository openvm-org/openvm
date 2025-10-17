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

#[derive(Debug)]
pub struct MerkleTree<F, const CHUNK: usize> {
    /// Height of the tree -- the root is the only node at height `height`,
    /// and the leaves are at height `0`.
    height: usize,
    /// Nodes corresponding to all zeroes.
    zero_nodes: Vec<[F; CHUNK]>,
    /// Nodes in the tree that have ever been touched.
    nodes: FxHashMap<u64, [F; CHUNK]>,
}

impl<F: PrimeField32, const CHUNK: usize> MerkleTree<F, CHUNK> {
    pub fn new(height: usize, hasher: &impl Hasher<CHUNK, F>) -> Self {
        Self {
            height,
            zero_nodes: (0..height + 1)
                .scan(hasher.hash(&[F::ZERO; CHUNK]), |acc, _| {
                    let result = Some(*acc);
                    *acc = hasher.compress(acc, acc);
                    result
                })
                .collect(),
            nodes: FxHashMap::default(),
        }
    }

    pub fn root(&self) -> [F; CHUNK] {
        self.get_node(1)
    }

    pub fn get_node(&self, index: u64) -> [F; CHUNK] {
        self.nodes
            .get(&index)
            .cloned()
            .unwrap_or(self.zero_nodes[self.height - index.ilog2() as usize])
    }

    #[allow(clippy::type_complexity)]
    /// Shared logic for both from_memory and finalize.
    fn process_layers<CompressFn>(
        &mut self,
        layer: Vec<(u64, [F; CHUNK])>,
        md: &MemoryDimensions,
        mut rows: Option<&mut Vec<MemoryMerkleCols<F, CHUNK>>>,
        compress: CompressFn,
    ) where
        CompressFn: Fn(&[F; CHUNK], &[F; CHUNK]) -> [F; CHUNK] + Send + Sync,
    {
        let mut new_entries = layer;
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
                            let left = if let Some(left) = left {
                                left.0
                            } else {
                                &self.get_node(2 * par_index)
                            };
                            let right = if let Some(right) = right {
                                right.0
                            } else {
                                &self.get_node(2 * par_index + 1)
                            };
                            let combined = compress(left, right);
                            let par_old_values = self.get_node(par_index);
                            (par_index, combined, par_old_values)
                        })
                        .collect();
                }
                Some(ref mut rows) => {
                    let label_section_height = md.address_height.saturating_sub(height);
                    let (tmp, new_rows): (Vec<(u64, [F; CHUNK], [F; CHUNK])>, Vec<[_; 2]>) =
                        new_layer
                            .into_par_iter()
                            .map(|(par_index, left, right)| {
                                let parent_address_label =
                                    (par_index & ((1 << label_section_height) - 1)) as u32;
                                let parent_as_label = ((par_index & !(1 << (self.height - height)))
                                    >> label_section_height)
                                    as u32;
                                let left_node;
                                let (left, old_left, changed_left) = match left {
                                    Some((left, old_left)) => (left, old_left, true),
                                    None => {
                                        left_node = self.get_node(2 * par_index);
                                        (&left_node, &left_node, false)
                                    }
                                };
                                let right_node;
                                let (right, old_right, changed_right) = match right {
                                    Some((right, old_right)) => (right, old_right, true),
                                    None => {
                                        right_node = self.get_node(2 * par_index + 1);
                                        (&right_node, &right_node, false)
                                    }
                                };
                                let combined = compress(left, right);
                                // This is a hacky way to say:
                                // "and we also want to record the old values"
                                compress(old_left, old_right);
                                let par_old_values = self.get_node(par_index);
                                (
                                    (par_index, combined, par_old_values),
                                    [
                                        MemoryMerkleCols {
                                            expand_direction: F::ONE,
                                            height_section: F::from_bool(
                                                height > md.address_height,
                                            ),
                                            parent_height: F::from_canonical_usize(height),
                                            is_root: F::from_bool(height == md.overall_height()),
                                            parent_as_label: F::from_canonical_u32(parent_as_label),
                                            parent_address_label: F::from_canonical_u32(
                                                parent_address_label,
                                            ),
                                            parent_hash: par_old_values,
                                            left_child_hash: *old_left,
                                            right_child_hash: *old_right,
                                            left_direction_different: F::ZERO,
                                            right_direction_different: F::ZERO,
                                        },
                                        MemoryMerkleCols {
                                            expand_direction: F::NEG_ONE,
                                            height_section: F::from_bool(
                                                height > md.address_height,
                                            ),
                                            parent_height: F::from_canonical_usize(height),
                                            is_root: F::from_bool(height == md.overall_height()),
                                            parent_as_label: F::from_canonical_u32(parent_as_label),
                                            parent_address_label: F::from_canonical_u32(
                                                parent_address_label,
                                            ),
                                            parent_hash: combined,
                                            left_child_hash: *left,
                                            right_child_hash: *right,
                                            left_direction_different: F::from_bool(!changed_left),
                                            right_direction_different: F::from_bool(!changed_right),
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

        if self.nodes.is_empty() {
            // This, for example, should happen in every `from_memory` call
            self.nodes = FxHashMap::from_iter(new_entries);
        } else {
            self.nodes.extend(new_entries);
        }
    }

    pub fn from_memory(
        memory: &AddressMap,
        md: &MemoryDimensions,
        hasher: &(impl Hasher<CHUNK, F> + Sync),
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
        hasher: &impl HasherChip<CHUNK, F>,
        touched: &Equipartition<F, CHUNK>,
        md: &MemoryDimensions,
    ) -> FinalState<CHUNK, F> {
        let init_root = self.get_node(1);
        let layer: Vec<_> = if !touched.is_empty() {
            touched
                .iter()
                .map(|((addr_sp, ptr), v)| {
                    (
                        (1 << self.height) + md.label_to_index((*addr_sp, *ptr / CHUNK as u32)),
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
}
