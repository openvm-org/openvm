use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::NBitXorCols, NBitXorChip};

impl<const N: usize, const M: usize> NBitXorChip<N, M> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = NBitXorCols::<N, M, F>::get_width();

        let mut rows = vec![];
        for (x, y) in self.pairs.iter() {
            let mut cols = NBitXorCols::<N, M, F>::from_placeholder(F::zero());
            cols.x = F::from_canonical_u32(*x);
            cols.y = F::from_canonical_u32(*y);

            let num_limbs = (N + M - 1) / M;

            let mut result = 0;

            let mut current_x = *x;
            let mut current_y = *y;
            for i in 0..num_limbs {
                let current_x_limb = current_x & ((1 << M) - 1);
                let current_y_limb = current_y & ((1 << M) - 1);

                let current_xor = self.mbit_xor_chip.request(current_x_limb, current_y_limb);
                result = result | (current_xor << (i * M));

                cols.x_limbs[i] = F::from_canonical_u32(current_x_limb);
                cols.y_limbs[i] = F::from_canonical_u32(current_y_limb);
                cols.z_limbs[i] = F::from_canonical_u32(current_xor);

                current_x >>= M;
                current_y >>= M;
            }

            cols.z = F::from_canonical_u32(result);

            let row = cols.flatten();

            rows.push(row);
        }

        RowMajorMatrix::new(rows.concat(), num_xor_cols)
    }
}
