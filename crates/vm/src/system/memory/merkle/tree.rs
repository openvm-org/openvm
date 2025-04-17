use openvm_stark_backend::p3_field::PrimeField32;
use rustc_hash::FxHashMap;

use super::{FinalState, MemoryMerkleCols};
use crate::{
    arch::hasher::HasherChip,
    system::memory::{
        dimensions::MemoryDimensions, memory_to_partition, Equipartition, MemoryImage,
    },
};

pub struct MerkleTree<F, const CHUNK: usize> {
    /// Height of the tree -- the root is the only node at depth 0, and the leaves are at depth
    /// `height`.
    height: usize,
    /// Nodes corresponding to all zeroes.
    zero_nodes: Vec<[F; CHUNK]>,
    /// Nodes in the tree that have ever been touched.
    nodes: FxHashMap<u64, [F; CHUNK]>,
    /// Actual values for the leaves (for the boundary chip).
    // We could create a special enum for the nodes, but I think this is cleaner.
    leaf_values: Option<Equipartition<F, CHUNK>>,
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
            leaf_values: None,
        }
    }

    pub fn from_memory<const FILL_LEAVES: bool>(
        image: &MemoryImage,
        md: &MemoryDimensions,
        hasher: &impl HasherChip<CHUNK, F>,
    ) -> Self {
        let mut tree = Self::new(md.overall_height(), hasher);
        tree.update::<FILL_LEAVES>(hasher, memory_to_partition(image), md, None);
        tree
    }

    pub fn with_leaf_values(self, leaf_values: Equipartition<F, CHUNK>) -> Self {
        Self {
            leaf_values: Some(leaf_values),
            ..self
        }
    }

    fn get_node(&self, index: u64) -> [F; CHUNK] {
        self.nodes
            .get(&index)
            .cloned()
            .unwrap_or(self.zero_nodes[self.height - index.ilog2() as usize])
    }

    pub fn get_leaf_value(&self, label: (u32, u32)) -> [F; CHUNK] {
        self.leaf_values
            .as_ref()
            .expect("Leaf values not initialized")
            .get(&label)
            .cloned()
            .unwrap_or([F::ZERO; CHUNK])
    }

    pub fn finalize<const UPDATE_LEAVES: bool>(
        &mut self,
        hasher: &mut impl HasherChip<CHUNK, F>,
        touched: Equipartition<F, CHUNK>,
        md: &MemoryDimensions,
    ) -> FinalState<CHUNK, F> {
        let mut rows = Vec::new();
        let init_root = self.get_node(1);
        self.update::<UPDATE_LEAVES>(hasher, touched, md, Some(&mut rows));
        FinalState {
            rows,
            init_root,
            final_root: self.get_node(1),
        }
    }

    fn update<const UPDATE_LEAVES: bool>(
        &mut self,
        hasher: &impl HasherChip<CHUNK, F>,
        touched: Equipartition<F, CHUNK>,
        md: &MemoryDimensions,
        mut rows: Option<&mut Vec<MemoryMerkleCols<F, CHUNK>>>,
    ) {
        let mut layer: Vec<_> = touched
            .iter()
            .map(|(k, v)| ((1 << self.height) + md.label_to_index(*k), *v))
            .collect();
        if UPDATE_LEAVES {
            if let Some(leaf_values) = &mut self.leaf_values {
                for (k, v) in touched.iter() {
                    leaf_values.insert(*k, *v);
                }
            } else {
                self.leaf_values = Some(touched);
            }
        }
        let rows_count = if layer.is_empty() {
            0
        } else {
            layer
                .iter()
                .zip(layer.iter().skip(1))
                .fold(md.overall_height(), |acc, ((lhs, _), (rhs, _))| {
                    acc + (lhs ^ rhs).ilog2() as usize
                })
        };
        if let Some(v) = rows.as_deref_mut() {
            v.reserve_exact(rows_count);
        }
        for height in 1..=self.height {
            let mut i = 0;
            let mut new_layer = Vec::new();
            while i < layer.len() {
                let (index, values) = layer[i];
                let par_index = index >> 1;
                i += 1;
                // Add the initial state record
                if let Some(v) = rows.as_deref_mut() {
                    v.push(MemoryMerkleCols {
                        expand_direction: F::ONE,
                        height_section: F::from_bool(height > md.address_height),
                        parent_height: F::from_canonical_usize(height),
                        is_root: F::from_bool(height == md.overall_height()),
                        parent_as_label: F::from_canonical_u32(
                            (par_index >> md.address_height) as u32,
                        ),
                        parent_address_label: F::from_canonical_u32(
                            (par_index & ((1 << md.address_height) - 1)) as u32,
                        ),
                        parent_hash: self.get_node(par_index),
                        left_child_hash: self.get_node(index),
                        right_child_hash: self.get_node(index + 1),
                        left_direction_different: F::ZERO,
                        right_direction_different: F::ZERO,
                    });
                }
                self.nodes.insert(index, values);
                if i < layer.len() && layer[i].0 == index ^ 1 {
                    // sibling found
                    let (_, sibling_values) = layer[i];
                    i += 1;
                    let combined = hasher.compress(&values, &sibling_values);
                    if let Some(v) = rows.as_deref_mut() {
                        v.push(MemoryMerkleCols {
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
                    }
                    self.nodes.insert(index + 1, sibling_values);
                    new_layer.push((par_index, combined));
                } else {
                    // no sibling found
                    let sibling_values = self.get_node(index ^ 1);
                    let is_left = index % 2 == 0;
                    let combined = hasher.compress(
                        if is_left { &values } else { &sibling_values },
                        if is_left { &sibling_values } else { &values },
                    );
                    if let Some(v) = rows.as_deref_mut() {
                        v.push(MemoryMerkleCols {
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
                    }
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
        if let Some(v) = rows {
            debug_assert_eq!(v.len(), rows_count);
        }
    }
}
