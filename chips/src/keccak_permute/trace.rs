use p3_field::PrimeField64;
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::keccak_permute::columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS};

use super::KeccakPermuteChip;

impl KeccakPermuteChip {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let keccak_trace: RowMajorMatrix<F> = generate_trace_rows(self.inputs.clone());

        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); keccak_trace.height() * NUM_KECCAK_PERMUTE_COLS],
            NUM_KECCAK_PERMUTE_COLS,
        );
        for i in 0..keccak_trace.height() {
            // TODO: Better way to do this, ideally the inner trace would be generated on &mut rows
            trace.row_mut(i)[..NUM_KECCAK_COLS].copy_from_slice(&keccak_trace.row_slice(i));
        }

        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakPermuteCols<F>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        for (i, row) in rows.iter_mut().enumerate() {
            if i < self.inputs.len() * NUM_ROUNDS {
                row.is_real = F::one();
                if i % NUM_ROUNDS == 0 {
                    row.is_real_input = F::one();
                }
                if i % NUM_ROUNDS == NUM_ROUNDS - 1 {
                    row.is_real_output = F::one();
                }
            }
        }

        trace
    }
}
