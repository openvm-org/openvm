use std::{
    borrow::BorrowMut,
    sync::{atomic::AtomicU32, Arc},
};

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use super::tree::MerkleTree;
use crate::{
    arch::hasher::HasherChip,
    system::{
        memory::{
            merkle::{FinalState, MemoryMerkleChip, MemoryMerkleCols},
            Equipartition,
        },
        poseidon2::{
            Poseidon2PeripheryBaseChip, Poseidon2PeripheryChip, PERIPHERY_POSEIDON2_WIDTH,
        },
    },
};

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    pub fn finalize<const UPDATE_LEAVES: bool>(
        &mut self,
        final_memory: Equipartition<F, CHUNK>,
        tree: &mut MerkleTree<F, CHUNK>,
        hasher: &mut impl HasherChip<CHUNK, F>,
    ) {
        self.final_state =
            Some(tree.finalize::<UPDATE_LEAVES>(hasher, final_memory, &self.air.memory_dimensions));
    }
}

impl<const CHUNK: usize, SC: StarkGenericConfig> Chip<SC> for MemoryMerkleChip<CHUNK, Val<SC>>
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
        // initial root must be the first, final root must be the second
        rows.reverse();
        if !rows.is_empty() {
            rows.swap(0, 1);
        }

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
impl<const CHUNK: usize, F: PrimeField32> ChipUsageGetter for MemoryMerkleChip<CHUNK, F> {
    fn air_name(&self) -> String {
        "Merkle".to_string()
    }

    fn current_trace_height(&self) -> usize {
        unimplemented!()
    }

    fn trace_width(&self) -> usize {
        MemoryMerkleCols::<F, CHUNK>::width()
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
