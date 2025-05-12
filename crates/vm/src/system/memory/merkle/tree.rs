use openvm_stark_backend::p3_field::PrimeField32;
use rustc_hash::FxHashMap;

use super::{memory_to_partition, FinalState, MemoryMerkleCols};
use crate::{
    arch::hasher::HasherChip,
    system::memory::{dimensions::MemoryDimensions, AddressMap, Equipartition, PAGE_SIZE},
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
    pub fn new(height: usize, hasher: &impl HasherChip<CHUNK, F>) -> Self {
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

    /// Shared logic for both from_memory and finalize.
    fn process_layers<CompressFn>(
        &mut self,
        layer: Vec<(u64, [F; CHUNK])>,
        md: &MemoryDimensions,
        mut rows: Option<&mut Vec<MemoryMerkleCols<F, CHUNK>>>,
        mut compress: CompressFn,
    ) where
        CompressFn: FnMut(&[F; CHUNK], &[F; CHUNK]) -> [F; CHUNK],
    {
        let mut layer = layer
            .into_iter()
            .map(|(index, values)| (index, values, self.get_node(index)))
            .collect::<Vec<_>>();
        for height in 1..=self.height {
            let mut i = 0;
            let mut new_layer = Vec::new();
            while i < layer.len() {
                let (index, values, old_values) = layer[i];
                let par_index = index >> 1;
                i += 1;

                let par_old_values = self.get_node(par_index);

                // Lowest `label_section_height` bits of `par_index` are the address label,
                // The remaining highest are the address space label.
                let label_section_height = md.address_height.saturating_sub(height);
                let parent_address_label = (par_index & ((1 << label_section_height) - 1)) as u32;
                let parent_as_label =
                    ((par_index & !(1 << (self.height - height))) >> label_section_height) as u32;

                self.nodes.insert(index, values);

                if i < layer.len() && layer[i].0 == index ^ 1 {
                    // sibling found
                    let (_, sibling_values, sibling_old_values) = layer[i];
                    i += 1;
                    let combined = compress(&values, &sibling_values);

                    // Only record rows if requested
                    if let Some(rows) = rows.as_deref_mut() {
                        rows.push(MemoryMerkleCols {
                            expand_direction: F::ONE,
                            height_section: F::from_bool(height > md.address_height),
                            parent_height: F::from_canonical_usize(height),
                            is_root: F::from_bool(height == md.overall_height()),
                            parent_as_label: F::from_canonical_u32(parent_as_label),
                            parent_address_label: F::from_canonical_u32(parent_address_label),
                            parent_hash: self.get_node(par_index),
                            left_child_hash: old_values,
                            right_child_hash: sibling_old_values,
                            left_direction_different: F::ZERO,
                            right_direction_different: F::ZERO,
                        });
                        rows.push(MemoryMerkleCols {
                            expand_direction: F::NEG_ONE,
                            height_section: F::from_bool(height > md.address_height),
                            parent_height: F::from_canonical_usize(height),
                            is_root: F::from_bool(height == md.overall_height()),
                            parent_as_label: F::from_canonical_u32(parent_as_label),
                            parent_address_label: F::from_canonical_u32(parent_address_label),
                            parent_hash: combined,
                            left_child_hash: values,
                            right_child_hash: sibling_values,
                            left_direction_different: F::ZERO,
                            right_direction_different: F::ZERO,
                        });
                        // This is a hacky way to say "and we also want to record the old values"
                        compress(&old_values, &sibling_old_values);
                    }

                    self.nodes.insert(index ^ 1, sibling_values);
                    new_layer.push((par_index, combined, par_old_values));
                } else {
                    // no sibling found
                    let sibling_values = self.get_node(index ^ 1);
                    let is_left = index % 2 == 0;
                    let (left, right) = if is_left {
                        (values, sibling_values)
                    } else {
                        (sibling_values, values)
                    };
                    let combined = compress(&left, &right);

                    if let Some(rows) = rows.as_deref_mut() {
                        rows.push(MemoryMerkleCols {
                            expand_direction: F::ONE,
                            height_section: F::from_bool(height > md.address_height),
                            parent_height: F::from_canonical_usize(height),
                            is_root: F::from_bool(height == md.overall_height()),
                            parent_as_label: F::from_canonical_u32(parent_as_label),
                            parent_address_label: F::from_canonical_u32(parent_address_label),
                            parent_hash: self.get_node(par_index),
                            left_child_hash: if is_left { old_values } else { left },
                            right_child_hash: if is_left { right } else { old_values },
                            left_direction_different: F::ZERO,
                            right_direction_different: F::ZERO,
                        });
                        rows.push(MemoryMerkleCols {
                            expand_direction: F::NEG_ONE,
                            height_section: F::from_bool(height > md.address_height),
                            parent_height: F::from_canonical_usize(height),
                            is_root: F::from_bool(height == md.overall_height()),
                            parent_as_label: F::from_canonical_u32(parent_as_label),
                            parent_address_label: F::from_canonical_u32(parent_address_label),
                            parent_hash: combined,
                            left_child_hash: left,
                            right_child_hash: right,
                            left_direction_different: F::from_bool(!is_left),
                            right_direction_different: F::from_bool(is_left),
                        });
                        // This is a hacky way to say "and we also want to record the old values"
                        if is_left {
                            compress(&old_values, &right);
                        } else {
                            compress(&left, &old_values);
                        }
                    }

                    new_layer.push((par_index, combined, par_old_values));
                }
            }
            layer = new_layer;
        }
        if !layer.is_empty() {
            assert_eq!(layer.len(), 1);
            self.nodes.insert(layer[0].0, layer[0].1);
        }
    }

    pub fn from_memory(
        initial_memory: AddressMap<PAGE_SIZE>,
        md: &MemoryDimensions,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> Self {
        let mut tree = Self::new(md.overall_height(), hasher);
        let layer: Vec<_> = memory_to_partition(&initial_memory)
            .iter()
            .map(|((addr_sp, ptr), v)| {
                (
                    (1 << tree.height) + md.label_to_index((*addr_sp, *ptr)),
                    hasher.hash(v),
                )
            })
            .collect();
        tree.process_layers(layer, md, None, |left, right| hasher.compress(left, right));
        tree
    }

    pub fn finalize(
        &mut self,
        hasher: &mut impl HasherChip<CHUNK, F>,
        touched: &Equipartition<F, CHUNK>,
        md: &MemoryDimensions,
    ) -> FinalState<CHUNK, F> {
        let init_root = self.get_node(1);
        let layer: Vec<_> = touched
            .iter()
            .map(|((addr_sp, ptr), v)| {
                (
                    (1 << self.height) + md.label_to_index((*addr_sp, *ptr / CHUNK as u32)),
                    hasher.hash(v),
                )
            })
            .collect();
        let mut rows = Vec::with_capacity(if touched.is_empty() {
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
        let final_root = self.get_node(1);
        FinalState {
            rows,
            init_root,
            final_root,
        }
    }

    fn get_node(&self, index: u64) -> [F; CHUNK] {
        self.nodes
            .get(&index)
            .cloned()
            .unwrap_or(self.zero_nodes[self.height - index.ilog2() as usize])
    }
}
