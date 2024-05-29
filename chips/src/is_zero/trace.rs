use super::columns::NUM_COLS;
use super::IsZeroChip;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

impl IsZeroChip {
    pub fn generate_trace_rows<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .x
            .iter()
            .map(|x| {
                let answer = if *x == 0 { 1 } else { 0 };
                let inv = F::from_canonical_u32(*x + answer).inverse();
                vec![
                    F::from_canonical_u32(*x),
                    F::from_canonical_u32(answer),
                    inv,
                ]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), NUM_COLS)
    }
}
