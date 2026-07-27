use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::{Arc, Mutex},
};

use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::LocalOpcode;
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::AirProvingContext,
    StarkProtocolConfig, Val,
};
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};

use crate::{
    keccakf_op::KeccakfPreimage,
    keccakf_perm::{KeccakfPermCols, NUM_KECCAKF_PERM_COLS},
};

#[derive(Clone, derive_new::new)]
pub struct KeccakfPermChip {
    pub(crate) shared_preimages: Arc<Mutex<Vec<KeccakfPreimage>>>,
}

impl<RA, SC> Chip<RA, CpuBackend<SC>> for KeccakfPermChip
where
    SC: StarkProtocolConfig,
    Val<SC>: PrimeField32,
{
    /// Generates the trace and clears the shared preimage handoff.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let preimages = std::mem::take(&mut *self.shared_preimages.lock().unwrap());
        AirProvingContext::simple_no_pis(generate_trace_from_preimages::<Val<SC>>(&preimages))
    }
}

/// Generates the Keccak permutation trace from preimages reconstructed by the Keccak operation
/// generator.
pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &KeccakfPermChip,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let preimages = std::mem::take(&mut *chip.shared_preimages.lock().unwrap());
    let expected_preimages = postflight
        .steps(KeccakfOpcode::KECCAKF.global_opcode())
        .len();
    if preimages.len() != expected_preimages {
        return Err(PostflightError::new(format!(
            "Keccak permutation expected {expected_preimages} reconstructed states, found {}",
            preimages.len()
        )));
    }
    Ok(generate_trace_from_preimages(&preimages))
}

fn generate_trace_from_preimages<F: PrimeField32>(
    preimages: &[KeccakfPreimage],
) -> RowMajorMatrix<F> {
    if preimages.is_empty() {
        return RowMajorMatrix::new(Vec::new(), NUM_KECCAKF_PERM_COLS);
    }
    let states = preimages
        .iter()
        .map(|preimage| {
            // p3-keccak-air now uses standard Keccak indexing:
            // input[x + 5*y] = state[x][y], matching the byte buffer layout.
            // The previous transposition workaround (plonky3 issue #672) is no longer needed.
            from_fn(|i| u64::from_le_bytes(preimage.bytes[i * 8..i * 8 + 8].try_into().unwrap()))
        })
        .collect::<Vec<_>>();

    let p3_trace = generate_trace_rows::<F>(states, 0);
    // Row-major: we need to add more columns
    let mut values = F::zero_vec(NUM_KECCAKF_PERM_COLS * p3_trace.height());
    values
        .par_chunks_exact_mut(NUM_KECCAKF_PERM_COLS)
        .zip(p3_trace.values.par_chunks_exact(NUM_KECCAK_COLS))
        .enumerate()
        .for_each(|(row_idx, (row, p3_row))| {
            row[..NUM_KECCAK_COLS].copy_from_slice(p3_row);

            if row_idx % NUM_ROUNDS == (NUM_ROUNDS - 1) {
                let preimage_idx = row_idx / NUM_ROUNDS;
                if let Some(preimage) = preimages.get(preimage_idx) {
                    let local: &mut KeccakfPermCols<_> = row.borrow_mut();
                    local.inner.export = F::ONE;
                    local.timestamp = F::from_u32(preimage.timestamp);
                }
            }
        });
    RowMajorMatrix::new(values, NUM_KECCAKF_PERM_COLS)
}
