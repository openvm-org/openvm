use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::{Arc, Mutex},
};

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{FieldAlgebra, PrimeField32},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip,
};
use p3_keccak_air::{generate_trace_rows, KeccakCols, NUM_KECCAK_COLS, NUM_ROUNDS};

use crate::keccakf_op::KeccakfRecord;

#[derive(Clone, derive_new::new)]
pub struct KeccakfPermChip {
    /// See comments in [KeccakfOpChip](crate::keccakf_op::KeccakfOpChip).
    pub(crate) shared_records: Arc<Mutex<Vec<KeccakfRecord>>>,
}

impl<RA, SC> Chip<RA, CpuBackend<SC>> for KeccakfPermChip
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
{
    /// Generates trace and clears internal records state.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let records: Vec<_> = std::mem::take(&mut self.shared_records.lock().unwrap());
        let states = records
            .iter()
            .map(|record| {
                // We need to transpose state matrices due to a plonky3 issue: https://github.com/Plonky3/Plonky3/issues/672
                // Note: the fix for this issue will be a commit after the major Field crate refactor PR https://github.com/Plonky3/Plonky3/pull/640
                //       which will require a significant refactor to switch to.
                from_fn(|i| {
                    let x = i / 5;
                    let y = i % 5;
                    let offset = x * 5 + y;
                    u64::from_le_bytes(
                        record.preimage_buffer_bytes[offset * 8..offset * 8 + 8]
                            .try_into()
                            .unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();

        let mut matrix = generate_trace_rows::<Val<SC>>(states, 0);
        // Set export flags
        for row_chunk in matrix
            .values
            .chunks_exact_mut(NUM_KECCAK_COLS * NUM_ROUNDS)
            .take(records.len())
        {
            let last_row = row_chunk.chunks_exact_mut(NUM_KECCAK_COLS).last().unwrap();
            let local: &mut KeccakCols<_> = last_row.borrow_mut();
            local.export = Val::<SC>::ONE;
        }

        AirProvingContext::simple_no_pis(Arc::new(matrix))
    }
}
