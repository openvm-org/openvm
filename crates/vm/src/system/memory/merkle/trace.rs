use std::{
    borrow::BorrowMut,
    sync::{atomic::AtomicU32, Arc},
};

use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_matrix::dense::RowMajorMatrix,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    ChipUsageGetter,
};
use tracing::instrument;

use crate::{
    arch::hasher::HasherChip,
    system::{
        memory::{
            merkle::{tree::MerkleTree, FinalState, MemoryMerkleChip, MemoryMerkleCols},
            Equipartition, MemoryImage,
        },
        poseidon2::{
            Poseidon2PeripheryBaseChip, Poseidon2PeripheryChip, PERIPHERY_POSEIDON2_WIDTH,
        },
    },
};

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    #[instrument(name = "merkle_finalize", level = "debug", skip_all)]
    pub(crate) fn finalize(
        &mut self,
        initial_memory: &MemoryImage,
        final_memory: &Equipartition<F, CHUNK>,
        hasher: &impl HasherChip<CHUNK, F>,
    ) {
        assert!(self.final_state.is_none(), "Merkle chip already finalized");
        let mut tree = MerkleTree::from_memory(initial_memory, &self.air.memory_dimensions, hasher);
        self.final_state = Some(tree.finalize(hasher, final_memory, &self.air.memory_dimensions));
    }
}

impl<const CHUNK: usize, F> MemoryMerkleChip<CHUNK, F>
where
    F: PrimeField32,
{
    pub fn generate_proving_ctx<SC>(&mut self) -> AirProvingContext<CpuBackend<SC>>
    where
        SC: StarkGenericConfig,
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        assert!(
            self.final_state.is_some(),
            "Merkle chip must finalize before trace generation"
        );
        let FinalState {
            mut rows,
            init_root,
            final_root,
        } = self.final_state.take().unwrap();
        // important that this sort be stable,
        // because we need the initial root to be first and the final root to be second
        rows.reverse();
        rows.swap(0, 1);

        #[cfg(feature = "metrics")]
        {
            self.current_height = rows.len();
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

        let trace = Arc::new(RowMajorMatrix::new(trace, width));
        let pvs = init_root.into_iter().chain(final_root).collect();
        AirProvingContext::simple(trace, pvs)
    }
}
impl<const CHUNK: usize, F: PrimeField32> ChipUsageGetter for MemoryMerkleChip<CHUNK, F> {
    fn air_name(&self) -> String {
        "Merkle".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.final_state.as_ref().map(|s| s.rows.len()).unwrap_or(0)
    }

    fn trace_width(&self) -> usize {
        MemoryMerkleCols::<F, CHUNK>::width()
    }
}

pub trait SerialReceiver<T> {
    fn receive(&self, msg: T);
}

impl<'a, F: PrimeField32, const SBOX_REGISTERS: usize> SerialReceiver<&'a [F]>
    for Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>
{
    /// Receives a permutation preimage, pads with zeros to the permutation width, and records.
    /// The permutation preimage must have length at most the permutation width (panics otherwise).
    fn receive(&self, perm_preimage: &'a [F]) {
        assert!(perm_preimage.len() <= PERIPHERY_POSEIDON2_WIDTH);
        let mut state = [F::ZERO; PERIPHERY_POSEIDON2_WIDTH];
        state[..perm_preimage.len()].copy_from_slice(perm_preimage);
        let count = self.records.entry(state).or_insert(AtomicU32::new(0));
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<'a, F: PrimeField32> SerialReceiver<&'a [F]> for Poseidon2PeripheryChip<F> {
    fn receive(&self, perm_preimage: &'a [F]) {
        match self {
            Poseidon2PeripheryChip::Register0(chip) => chip.receive(perm_preimage),
            Poseidon2PeripheryChip::Register1(chip) => chip.receive(perm_preimage),
        }
    }
}
