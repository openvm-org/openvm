use ax_stark_backend::p3_field::PrimeField32;
use ax_stark_sdk::dummy_airs::fib_air::columns::NUM_FIBONACCI_COLS;
use p3_matrix::dense::RowMajorMatrix;

/// sels contain boolean selectors to enable the fibonacci gate
pub fn generate_trace_rows<F: PrimeField32>(a: u32, b: u32, sels: &[bool]) -> RowMajorMatrix<F> {
    let n = sels.len();
    assert!(n.is_power_of_two());

    let mut rows = vec![vec![F::from_canonical_u32(a), F::from_canonical_u32(b)]];

    for i in 1..n {
        if sels[i - 1] {
            rows.push(vec![rows[i - 1][1], rows[i - 1][0] + rows[i - 1][1]]);
        } else {
            rows.push(vec![rows[i - 1][0], rows[i - 1][1]]);
        }
    }

    RowMajorMatrix::new(rows.concat(), NUM_FIBONACCI_COLS)
}
