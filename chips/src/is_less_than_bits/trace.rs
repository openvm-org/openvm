use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::{IsLessThanBitsAuxCols, IsLessThanBitsCols, IsLessThanBitsIoCols};
use super::IsLessThanBitsAir;
use crate::sub_chip::LocalTraceInstructions;

impl IsLessThanBitsAir {
    pub fn generate_trace<F: PrimeField64>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        let num_cols: usize = IsLessThanBitsCols::<F>::get_width(self.limb_bits);

        let mut rows = vec![];

        // generate a row for each pair of numbers to compare
        for (x, y) in pairs {
            let row: Vec<F> = self.generate_trace_row((x, y)).flatten();
            rows.extend(row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for IsLessThanBitsAir {
    type LocalInput = (u32, u32);

    fn generate_trace_row(&self, input: (u32, u32)) -> Self::Cols<F> {
        let (x, y) = input;

        let source = (1 << self.limb_bits) + x - y;
        let mut source_bits = vec![];
        for d in 0..=self.limb_bits {
            let source_bit = (source >> d) & 1;
            source_bits.push(F::from_canonical_u32(source_bit));
        }

        let io = IsLessThanBitsIoCols {
            x: F::from_canonical_u32(x),
            y: F::from_canonical_u32(y),
            is_less_than: F::from_bool(x < y),
        };
        let aux = IsLessThanBitsAuxCols { source_bits };

        IsLessThanBitsCols { io, aux }
    }
}
