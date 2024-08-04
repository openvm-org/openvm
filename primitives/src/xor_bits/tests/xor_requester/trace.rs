use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::NUM_XOR_REQUESTER_COLS, XorRequesterChip};

impl<const N: usize> XorRequesterChip<N> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for (x, y) in self.requests.iter() {
            let mut row = vec![];

            let z = self.xor_chip.request(*x, *y);

            row.push(F::from_canonical_u32(*x));
            row.push(F::from_canonical_u32(*y));
            row.push(F::from_canonical_u32(z));

            rows.push(row);
        }

        RowMajorMatrix::new(rows.concat(), NUM_XOR_REQUESTER_COLS)
    }
}
