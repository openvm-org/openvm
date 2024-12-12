use openvm_stark_backend::p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::fib_triples_air::columns::NUM_FIBONACCI_COLS;

/// n is number of rows in the trace
pub fn generate_trace_rows<F: PrimeField32>(a: u32, b: u32, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let c = a + b;

    let mut rows = vec![vec![
        F::from_canonical_u32(a),
        F::from_canonical_u32(b),
        F::from_canonical_u32(c),
    ]];

    for i in 1..n {
        rows.push(vec![
            rows[i - 1][1],
            rows[i - 1][2],
            rows[i - 1][1] + rows[i - 1][2],
        ]);
    }

    RowMajorMatrix::new(rows.concat(), NUM_FIBONACCI_COLS)
}
