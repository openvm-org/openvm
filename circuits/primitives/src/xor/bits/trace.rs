use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{XorCols, XorColsMut},
    XorBitsAir, XorBitsChip,
};

impl<const N: usize> XorBitsChip<N> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = XorCols::<N, F>::get_width();

        let mut pairs_locked = self.pairs.lock();
        pairs_locked.sort();

        let mut rows_concat = vec![F::zero(); num_xor_cols * pairs_locked.len()];
        for (i, (x, y)) in pairs_locked.iter().enumerate() {
            let xor_cols: XorColsMut<N, F> =
                XorColsMut::from_slice(&mut rows_concat[i * num_xor_cols..(i + 1) * num_xor_cols]);

            self.air.generate_trace_row(*x, *y, xor_cols);
        }

        RowMajorMatrix::new(rows_concat, num_xor_cols)
    }
}

impl<const N: usize> XorBitsAir<N> {
    fn generate_trace_row<F: AbstractField>(&self, x: u32, y: u32, xor_cols: XorColsMut<N, F>) {
        let z = self.calc_xor(x, y);

        [*xor_cols.io.x, *xor_cols.io.y, *xor_cols.io.z] = [x, y, z].map(F::from_canonical_u32);

        for i in 0..N {
            xor_cols.bits.x[i] = F::from_canonical_u32((x >> i) & 1);
            xor_cols.bits.y[i] = F::from_canonical_u32((y >> i) & 1);
            xor_cols.bits.z[i] = F::from_canonical_u32((z >> i) & 1);
        }
    }
}
