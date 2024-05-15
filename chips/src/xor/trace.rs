use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::XorCols, XorChip};

impl<const N: usize> XorChip<N> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = XorCols::<N, F>::get_width();

        let mut rows = vec![];
        for (x, y) in self.pairs.iter() {
            let mut cols = XorCols::<N, F>::from_placeholder(F::default());

            let z = self.calc_xor(*x, *y);

            cols.io.x = F::from_canonical_u32(*x);
            cols.io.y = F::from_canonical_u32(*y);
            cols.io.z = F::from_canonical_u32(z);

            for i in 0..N {
                cols.x_bits[i] = F::from_canonical_u32((x >> i) & 1);
                cols.y_bits[i] = F::from_canonical_u32((y >> i) & 1);
                cols.z_bits[i] = F::from_canonical_u32((z >> i) & 1);
            }

            let row = cols.flatten();

            rows.push(row);
        }

        RowMajorMatrix::new(rows.concat(), num_xor_cols)
    }
}
