use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::XorLimbsCols, XorLimbsChip};

impl<const N: usize, const M: usize> XorLimbsChip<N, M> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = XorLimbsCols::<N, M, F>::get_width();

        let mut pairs_locked = self.pairs.lock();
        pairs_locked.sort();

        let num_limbs = (N + M - 1) / M;

        let rows = pairs_locked
            .iter()
            .map(|(x, y)| {
                let z = self.calc_xor(*x, *y);

                let mut row = vec![
                    F::from_canonical_u32(*x),
                    F::from_canonical_u32(*y),
                    F::from_canonical_u32(z),
                ];

                let mut x_limbs = vec![];
                let mut y_limbs = vec![];
                let mut z_limbs = vec![];
                for i in 0..num_limbs {
                    let x_cur = (x >> (i * M)) & ((1 << M) - 1);
                    let y_cur = (y >> (i * M)) & ((1 << M) - 1);
                    let z_cur = (z >> (i * M)) & ((1 << M) - 1);

                    self.xor_lookup_chip.request(x_cur, y_cur);

                    x_limbs.push(F::from_canonical_u32(x_cur));
                    y_limbs.push(F::from_canonical_u32(y_cur));
                    z_limbs.push(F::from_canonical_u32(z_cur));
                }

                row.extend(x_limbs);
                row.extend(y_limbs);
                row.extend(z_limbs);

                row
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), num_xor_cols)
    }
}
