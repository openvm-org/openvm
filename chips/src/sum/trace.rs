use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::NUM_SUM_GATE_COLS;

pub fn generate_trace<F: PrimeField64>(inputs: &[u64]) -> RowMajorMatrix<F> {
    let n = inputs.len();

    let mut rows = Vec::with_capacity(n);
    let mut partial_sum = F::from_canonical_u64(0);

    for &input in inputs {
        let input_f = F::from_canonical_u64(input);
        partial_sum += input_f;
        rows.push(vec![input_f, partial_sum]);
    }

    RowMajorMatrix::new(rows.concat(), NUM_SUM_GATE_COLS)
}
