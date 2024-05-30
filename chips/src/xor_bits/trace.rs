use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::SubAirLocalTraceInstructions;

use super::{
    columns::{XorCols, XorIOCols},
    XorBitsChip,
};

impl<const N: usize> XorBitsChip<N> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let num_xor_cols: usize = XorCols::<N, F>::get_width();

        let mut pairs_locked = self.pairs.lock();
        pairs_locked.sort();

        let rows = pairs_locked
            .iter()
            .flat_map(|(x, y)| self.generate_trace_row((*x, *y)).flatten())
            .collect();

        RowMajorMatrix::new(rows, num_xor_cols)
    }
}

impl<const N: usize, F: AbstractField> SubAirLocalTraceInstructions<F> for XorBitsChip<N> {
    /// The input is (x, y) to be XOR-ed.
    type LocalInput = (u32, u32);

    fn generate_trace_row(&self, (x, y): (u32, u32)) -> Self::Cols<F> {
        let z = self.calc_xor(x, y);
        let [x_bits, y_bits, z_bits] = [x, y, z].map(|x| {
            (0..N)
                .map(|i| (x >> i) & 1)
                .map(F::from_canonical_u32)
                .collect()
        });
        let [x, y, z] = [x, y, z].map(F::from_canonical_u32);

        XorCols {
            io: XorIOCols { x, y, z },
            x_bits,
            y_bits,
            z_bits,
        }
    }
}
