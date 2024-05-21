use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::XorCols, XorBitsChip};

impl<const N: usize> XorBitsChip<N> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = XorCols::<N, F>::get_width();

        let mut pairs_locked = self.pairs.lock();
        pairs_locked.sort();

        let rows = pairs_locked
            .iter()
            .map(|(x, y)| {
                let z = self.calc_xor(*x, *y);

                let mut row = vec![
                    F::from_canonical_u32(*x),
                    F::from_canonical_u32(*y),
                    F::from_canonical_u32(z),
                ];

                row.extend((0..N).map(|i| (x >> i) & 1).map(F::from_canonical_u32));
                row.extend((0..N).map(|i| (y >> i) & 1).map(F::from_canonical_u32));
                row.extend((0..N).map(|i| (z >> i) & 1).map(F::from_canonical_u32));

                row
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), num_xor_cols)
    }
}
