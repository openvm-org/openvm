use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

const NUM_IS_ZERO_COLS: usize = 3; // Assuming a constant value for the number of columns

pub fn generate_trace_rows<F: PrimeField32>(x: u32) -> RowMajorMatrix<F> {
    let answer = if x == 0 { 1 } else { 0 };
    let inv = F::from_canonical_u32(x + answer).inverse();
    let rows = vec![F::from_canonical_u32(x), F::from_canonical_u32(answer), inv];

    RowMajorMatrix::new(rows, NUM_IS_ZERO_COLS)
}
