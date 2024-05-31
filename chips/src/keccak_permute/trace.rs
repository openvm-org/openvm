use p3_field::PrimeField64;
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    columns::{KECCAK_PERMUTE_COL_MAP, NUM_KECCAK_PERMUTE_COLS},
    KeccakPermuteChip,
};

impl KeccakPermuteChip {
    pub fn generate_trace<F: PrimeField64>(&self, inputs: Vec<[u64; 25]>) -> RowMajorMatrix<F> {
        let num_inputs = inputs.len();
        let keccak_trace: RowMajorMatrix<F> = generate_trace_rows(inputs);

        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); keccak_trace.height() * NUM_KECCAK_PERMUTE_COLS],
            NUM_KECCAK_PERMUTE_COLS,
        );
        for i in 0..keccak_trace.height() {
            // TODO: Better way to do this, ideally the inner trace would be generated on &mut rows
            trace.row_mut(i)[..NUM_KECCAK_COLS].copy_from_slice(&keccak_trace.row_slice(i));
        }

        for (i, row) in trace.rows_mut().enumerate() {
            if i < num_inputs * NUM_ROUNDS {
                row[KECCAK_PERMUTE_COL_MAP.is_real] = F::one();
                if i % NUM_ROUNDS == 0 {
                    row[KECCAK_PERMUTE_COL_MAP.is_real_input] = F::one();
                }
                if i % NUM_ROUNDS == NUM_ROUNDS - 1 {
                    row[KECCAK_PERMUTE_COL_MAP.is_real_output] = F::one();
                }
            }
        }

        trace
    }
}
