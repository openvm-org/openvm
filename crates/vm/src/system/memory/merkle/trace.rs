use std::{
    borrow::BorrowMut,
    cmp::Reverse,
    sync::{atomic::AtomicU32, Arc},
};

use crossbeam::{
    channel::{self, Sender},
    thread::scope,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};
use rustc_hash::FxHashSet;

use crate::{
    arch::hasher::Hasher,
    system::{
        memory::{
            controller::dimensions::MemoryDimensions,
            merkle::{FinalState, MemoryMerkleChip, MemoryMerkleCols},
            tree::MemoryNode::{self, NonLeaf},
            Equipartition,
        },
        poseidon2::{
            Poseidon2PeripheryBaseChip, Poseidon2PeripheryChip, PERIPHERY_POSEIDON2_WIDTH,
        },
    },
};

impl<F: PrimeField32, const CHUNK: usize> MemoryMerkleChip<CHUNK, F> {
    /// SAFETY: the `hash` implementation of `H: Hasher` must be pure, i.e., it does not depend on
    /// any mutable state in `hasher`. In particular the state mutations from `SerialReceiver` must not
    /// affect the functionality of the hash function.
    pub fn finalize<H>(
        &mut self,
        initial_tree: &MemoryNode<CHUNK, F>,
        final_memory: &Equipartition<F, CHUNK>,
        hasher: &mut H,
    ) where
        H: Hasher<CHUNK, F> + for<'a> SerialReceiver<&'a [F]> + Send + Sync,
    {
        assert!(self.final_state.is_none(), "Merkle chip already finalized");
        // there needs to be a touched node with `height_section` = 0
        // shouldn't be a leaf because
        // trace generation will expect an interaction from MemoryInterfaceChip in that case
        if self.touched_nodes.len() == 1 {
            self.touch_node(1, 0, 0);
        }

        let mut final_trace_rows = Vec::new();
        let final_tree = scope(|s| {
            let (trace_send, trace_recv) = channel::unbounded::<MemoryMerkleCols<F, CHUNK>>();
            // SAFETY: the only state changes to `hasher` are in records, which do no affect
            // the pure hash function
            let immutable_hasher = unsafe { &*(hasher as *const H) };
            let final_tree_handler = s.spawn(|_| {
                let memory_dimensions = self.air.memory_dimensions;
                let tree_helper = TreeHelper {
                    memory_dimensions,
                    final_memory,
                    touched_nodes: &self.touched_nodes,
                };
                tree_helper.recur(
                    memory_dimensions.overall_height(),
                    initial_tree,
                    0,
                    0,
                    immutable_hasher,
                    trace_send,
                )
            });

            // The loop breaks when last sender has been dropped,
            // so `recur` should be done
            while let Ok(row) = trace_recv.recv() {
                hasher.receive(row.hash_preimage());
                final_trace_rows.push(row);
            }
            final_tree_handler.join().unwrap()
        })
        .unwrap();
        self.final_state = Some(FinalState {
            rows: final_trace_rows,
            init_root: initial_tree.hash(),
            final_root: final_tree.hash(),
        });
    }
}

impl<SC: StarkGenericConfig, const CHUNK: usize> Chip<SC> for MemoryMerkleChip<CHUNK, Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        assert!(
            self.final_state.is_some(),
            "Merkle chip must finalize before trace generation"
        );
        let FinalState {
            mut rows,
            init_root,
            final_root,
        } = self.final_state.unwrap();
        // important that this sort be stable,
        // because we need the initial root to be first and the final root to be second
        rows.sort_by_key(|row| Reverse(row.parent_height));

        let width = MemoryMerkleCols::<Val<SC>, CHUNK>::width();
        let mut height = rows.len().next_power_of_two();
        if let Some(mut oh) = self.overridden_height {
            oh = oh.next_power_of_two();
            assert!(
                oh >= height,
                "Overridden height {oh} is less than the required height {height}"
            );
            height = oh;
        }
        let mut trace = Val::<SC>::zero_vec(width * height);

        for (trace_row, row) in trace.chunks_exact_mut(width).zip(rows) {
            *trace_row.borrow_mut() = row;
        }

        let trace = RowMajorMatrix::new(trace, width);
        let pvs = init_root.into_iter().chain(final_root).collect();
        AirProofInput::simple(trace, pvs)
    }
}
impl<F: PrimeField32, const CHUNK: usize> ChipUsageGetter for MemoryMerkleChip<CHUNK, F> {
    fn air_name(&self) -> String {
        "Merkle".to_string()
    }

    fn current_trace_height(&self) -> usize {
        2 * self.num_touched_nonleaves
    }

    fn trace_width(&self) -> usize {
        MemoryMerkleCols::<F, CHUNK>::width()
    }
}

struct TreeHelper<'a, F: PrimeField32, const CHUNK: usize> {
    memory_dimensions: MemoryDimensions,
    final_memory: &'a Equipartition<F, CHUNK>,
    touched_nodes: &'a FxHashSet<(usize, u32, u32)>,
}

// Heuristic: after the tree is less than this height, don't use rayon to multithread recursion.
const RAYON_JOIN_HEIGHT_THRESHOLD: usize = 12;

impl<F: PrimeField32, const CHUNK: usize> TreeHelper<'_, F, CHUNK> {
    // A divide-and-conquer recursion that spins up new threads for each recursive call.
    #[allow(clippy::too_many_arguments)]
    fn recur(
        &self,
        height: usize,
        initial_node: &MemoryNode<CHUNK, F>,
        as_label: u32,
        address_label: u32,
        hasher: &(impl Hasher<CHUNK, F> + Send + Sync),
        trace_send: Sender<MemoryMerkleCols<F, CHUNK>>,
    ) -> MemoryNode<CHUNK, F> {
        if height == 0 {
            let address_space = as_label + self.memory_dimensions.as_offset;
            let leaf_values = *self
                .final_memory
                .get(&(address_space, address_label))
                .unwrap_or(&[F::ZERO; CHUNK]);
            MemoryNode::new_leaf(hasher.hash(&leaf_values))
        } else if let NonLeaf {
            left: initial_left_node,
            right: initial_right_node,
            ..
        } = initial_node.clone()
        {
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
            let right_is_final =
                !self
                    .touched_nodes
                    .contains(&(height - 1, right_as_label, right_address_label));
            let recur_left = || {
                Arc::new(self.recur(
                    height - 1,
                    &initial_left_node,
                    left_as_label,
                    left_address_label,
                    hasher,
                    trace_send.clone(),
                ))
            };
            let recur_right = || {
                Arc::new(self.recur(
                    height - 1,
                    &initial_right_node,
                    right_as_label,
                    right_address_label,
                    hasher,
                    trace_send.clone(),
                ))
            };

            let (final_left_node, final_right_node) = match (left_is_final, right_is_final) {
                (true, true) => (initial_left_node, initial_right_node),
                (true, false) => (initial_left_node, recur_right()),
                (false, true) => (recur_left(), initial_right_node),
                (false, false) => {
                    if height > RAYON_JOIN_HEIGHT_THRESHOLD {
                        // Use rayon to run the child recursions potentially in different threads in the pool
                        join(recur_left, recur_right)
                    } else {
                        (recur_left(), recur_right())
                    }
                }
            };

            let final_node = MemoryNode::new_nonleaf(final_left_node, final_right_node, hasher);
            trace_send
                .send(self.trace_row(height, as_label, address_label, initial_node, None))
                .unwrap();
            let trace_row2 = self.trace_row(
                height,
                as_label,
                address_label,
                &final_node,
                Some([left_is_final, right_is_final]),
            );
            trace_send.send(trace_row2).unwrap();
            drop(trace_send);
            final_node
        } else {
            panic!("Leaf {:?} found at nonzero height {}", initial_node, height);
        }
    }

    /// Expects `node` to be NonLeaf
    fn trace_row(
        &self,
        parent_height: usize,
        as_label: u32,
        address_label: u32,
        node: &MemoryNode<CHUNK, F>,
        direction_changes: Option<[bool; 2]>,
    ) -> MemoryMerkleCols<F, CHUNK> {
        let [left_direction_change, right_direction_change] =
            direction_changes.unwrap_or([false; 2]);

        if let NonLeaf { hash, left, right } = node {
            MemoryMerkleCols {
                expand_direction: if direction_changes.is_none() {
                    F::ONE
                } else {
                    F::NEG_ONE
                },
                height_section: F::from_bool(parent_height > self.memory_dimensions.address_height),
                parent_height: F::from_canonical_usize(parent_height),
                is_root: F::from_bool(parent_height == self.memory_dimensions.overall_height()),
                parent_as_label: F::from_canonical_u32(as_label),
                parent_address_label: F::from_canonical_u32(address_label),
                parent_hash: *hash,
                left_child_hash: left.hash(),
                right_child_hash: right.hash(),
                left_direction_different: F::from_bool(left_direction_change),
                right_direction_different: F::from_bool(right_direction_change),
            }
        } else {
            panic!("trace_rows expects node = {:?} to be NonLeaf", node);
        }
    }
}

impl<T, const CHUNK: usize> MemoryMerkleCols<T, CHUNK> {
    /// Returns concatenation of `left_child_hash` and `right_child_hash` as a slice.
    fn hash_preimage(&self) -> &[T] {
        let ptr = &self.left_child_hash as *const T;
        // SAFETY: `MemoryMerkleCols` is repr(C) and `left_child_hash` and `right_child_hash` are
        // adjacent in memory. Therefore they can be concatenated into a slice.
        unsafe { &*std::ptr::slice_from_raw_parts(ptr, CHUNK * 2) }
    }
}

pub trait SerialReceiver<T> {
    fn receive(&mut self, msg: T);
}

impl<'a, F: PrimeField32, const SBOX_REGISTERS: usize> SerialReceiver<&'a [F]>
    for Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>
{
    /// Receives a permutation preimage, pads with zeros to the permutation width, and records.
    /// The permutation preimage must have length at most the permutation width (panics otherwise).
    fn receive(&mut self, perm_preimage: &'a [F]) {
        assert!(perm_preimage.len() <= PERIPHERY_POSEIDON2_WIDTH);
        let mut state = [F::ZERO; PERIPHERY_POSEIDON2_WIDTH];
        state[..perm_preimage.len()].copy_from_slice(perm_preimage);
        let count = self.records.entry(state).or_insert(AtomicU32::new(0));
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<'a, F: PrimeField32> SerialReceiver<&'a [F]> for Poseidon2PeripheryChip<F> {
    fn receive(&mut self, perm_preimage: &'a [F]) {
        match self {
            Poseidon2PeripheryChip::Register0(chip) => chip.receive(perm_preimage),
            Poseidon2PeripheryChip::Register1(chip) => chip.receive(perm_preimage),
        }
    }
}
