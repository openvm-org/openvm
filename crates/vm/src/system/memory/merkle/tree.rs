use openvm_stark_backend::p3_field::PrimeField32;
use rustc_hash::FxHashMap;

use super::{FinalState, MemoryMerkleCols};
use crate::{
    arch::hasher::HasherChip,
    system::memory::{dimensions::MemoryDimensions, Equipartition},
};

pub struct MerkleTree<F, const CHUNK: usize> {
    /// Height of the tree -- the root is the only node at depth 0, and the leaves are at depth
    /// `height`.
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
            zero_nodes: (0..height)
                .scan(hasher.hash(&[F::ZERO; CHUNK]), |acc, _| {
                    *acc = hasher.compress(acc, acc);
                    Some(*acc)
                })
                .collect(),
            nodes: FxHashMap::default(),
        }
    }

    fn get_node(&self, index: u64) -> [F; CHUNK] {
        self.nodes
            .get(&index)
            .cloned()
            .unwrap_or(self.zero_nodes[self.height - index.ilog2() as usize])
    }

    pub fn finalize(
        &mut self,
        hasher: &mut impl HasherChip<CHUNK, F>,
        touched: &Equipartition<F, CHUNK>,
        md: &MemoryDimensions,
    ) -> FinalState<CHUNK, F> {
        let mut layer: Vec<_> = touched
            .iter()
            .map(|(k, v)| ((1 << self.height) + md.label_to_index(*k), *v))
            .collect();
        let init_root = self.get_node(1);
        let mut rows = Vec::new();
        for height in 1..=self.height {
            let mut i = 0;
            let mut new_layer = Vec::new();
            while i < layer.len() {
                let (index, values) = layer[i];
                let par_index = index >> 1;
                i += 1;
                // Add the initial state record
                rows.push(MemoryMerkleCols {
                    expand_direction: F::ONE,
                    height_section: F::from_bool(height > md.address_height),
                    parent_height: F::from_canonical_usize(height),
                    is_root: F::from_bool(height == md.overall_height()),
                    parent_as_label: F::from_canonical_u32((par_index >> md.address_height) as u32),
                    parent_address_label: F::from_canonical_u32(
                        (par_index & ((1 << md.address_height) - 1)) as u32,
                    ),
                    parent_hash: self.get_node(par_index),
                    left_child_hash: self.get_node(index),
                    right_child_hash: self.get_node(index + 1),
                    left_direction_different: F::ZERO,
                    right_direction_different: F::ZERO,
                });
                self.nodes.insert(index, values);
                if i < layer.len() && layer[i].0 == index ^ 1 {
                    // sibling found
                    let (_, sibling_values) = layer[i];
                    i += 1;
                    let combined = hasher.compress_and_record(&values, &sibling_values);
                    rows.push(MemoryMerkleCols {
                        expand_direction: F::NEG_ONE,
                        height_section: F::from_bool(height > md.address_height),
                        parent_height: F::from_canonical_usize(height),
                        is_root: F::from_bool(height == md.overall_height()),
                        parent_as_label: F::from_canonical_u32(
                            (par_index >> md.address_height) as u32,
                        ),
                        parent_address_label: F::from_canonical_u32(
                            (par_index & ((1 << md.address_height) - 1)) as u32,
                        ),
                        parent_hash: combined,
                        left_child_hash: values,
                        right_child_hash: sibling_values,
                        left_direction_different: F::ONE,
                        right_direction_different: F::ONE,
                    });
                    self.nodes.insert(index + 1, sibling_values);
                    new_layer.push((par_index, combined));
                } else {
                    // no sibling found
                    let sibling_values = self.get_node(index ^ 1);
                    let is_left = index % 2 == 0;
                    let combined = hasher.compress_and_record(
                        if is_left { &values } else { &sibling_values },
                        if is_left { &sibling_values } else { &values },
                    );
                    rows.push(MemoryMerkleCols {
                        expand_direction: F::NEG_ONE,
                        height_section: F::from_bool(height > md.address_height),
                        parent_height: F::from_canonical_usize(height),
                        is_root: F::from_bool(height == md.overall_height()),
                        parent_as_label: F::from_canonical_u32(
                            (par_index >> md.address_height) as u32,
                        ),
                        parent_address_label: F::from_canonical_u32(
                            (par_index & ((1 << md.address_height) - 1)) as u32,
                        ),
                        parent_hash: combined,
                        left_child_hash: if is_left { values } else { sibling_values },
                        right_child_hash: if is_left { sibling_values } else { values },
                        left_direction_different: F::from_bool(is_left),
                        right_direction_different: F::from_bool(!is_left),
                    });
                    self.nodes.insert(index + 1, sibling_values);
                    new_layer.push((par_index, combined));
                }
            }
            layer = new_layer;
        }
        if !layer.is_empty() {
            assert_eq!(layer.len(), 1);
            self.nodes.insert(layer[0].0, layer[0].1);
        }
        FinalState {
            rows,
            init_root,
            final_root: self.get_node(1),
        }
    }
}
