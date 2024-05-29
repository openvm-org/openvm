use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::NUM_COLS, IsEqualChip};

impl IsEqualChip {
    pub fn generate_trace_rows<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .x
            .iter()
            .enumerate()
            .map(|(i, _x)| {
                let x = F::from_canonical_u32(*_x);
                let y = F::from_canonical_u32(self.y[i]);
                let answer = if x == y { F::one() } else { F::zero() };
                let inv = (x - y + answer).inverse();
                vec![x, y, answer, inv]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), NUM_COLS)
    }
}
