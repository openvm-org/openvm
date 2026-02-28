use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::{Arc, Mutex},
};

use openvm_circuit_primitives::Chip;
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    StarkProtocolConfig, Val,
};
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};

use crate::{
    keccakf_op::KeccakfRecord,
    keccakf_perm::{KeccakfPermCols, NUM_KECCAKF_PERM_COLS},
};

#[derive(Clone, derive_new::new)]
pub struct KeccakfPermChip {
    /// See comments in [KeccakfOpChip](crate::keccakf_op::KeccakfOpChip).
    pub(crate) shared_records: Arc<Mutex<Vec<KeccakfRecord>>>,
}

impl<RA, SC> Chip<RA, CpuBackend<SC>> for KeccakfPermChip
where
    SC: StarkProtocolConfig,
    Val<SC>: PrimeField32,
{
    /// Generates trace and clears internal records state.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let records: Vec<_> = std::mem::take(&mut self.shared_records.lock().unwrap());
        let states = records
            .iter()
            .map(|record| {
                // p3-keccak-air now uses standard Keccak indexing:
                // input[x + 5*y] = state[x][y], matching the byte buffer layout.
                // The previous transposition workaround (plonky3 issue #672) is no longer needed.
                from_fn(|i| {
                    u64::from_le_bytes(
                        record.preimage_buffer_bytes[i * 8..i * 8 + 8]
                            .try_into()
                            .unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();

        let p3_trace = generate_trace_rows::<Val<SC>>(states, 0);
        // Row-major: we need to add more columns
        let mut values = Val::<SC>::zero_vec(NUM_KECCAKF_PERM_COLS * p3_trace.height());
        values
            .par_chunks_exact_mut(NUM_KECCAKF_PERM_COLS)
            .zip(p3_trace.values.par_chunks_exact(NUM_KECCAK_COLS))
            .enumerate()
            .for_each(|(row_idx, (row, p3_row))| {
                row[..NUM_KECCAK_COLS].copy_from_slice(p3_row);

                if row_idx % NUM_ROUNDS == (NUM_ROUNDS - 1) {
                    let record_idx = row_idx / NUM_ROUNDS;
                    if let Some(record) = records.get(record_idx) {
                        let local: &mut KeccakfPermCols<_> = row.borrow_mut();
                        local.inner.export = Val::<SC>::ONE;
                        local.timestamp = Val::<SC>::from_u32(record.timestamp);
                    }
                }
            });
        let matrix = RowMajorMatrix::new(values, NUM_KECCAKF_PERM_COLS);
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&matrix))
    }
}
