use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::{IsLessThanBitsAuxCols, IsLessThanBitsCols, IsLessThanBitsIOCols};
use super::IsLessThanBitsAir;
use super::IsLessThanBitsChip;
use crate::sub_chip::LocalTraceInstructions;

impl IsLessThanBitsChip {
    pub fn generate_trace<F: PrimeField64>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        let num_cols: usize = IsLessThanBitsCols::<F>::get_width(self.air.limb_bits);

        let mut rows = vec![];

        // generate a row for each pair of numbers to compare
        for (x, y) in pairs {
            let row: Vec<F> = self.air.generate_trace_row((x, y)).flatten();
            rows.extend(row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for IsLessThanBitsAir {
    type LocalInput = (u32, u32);

    fn generate_trace_row(&self, input: (u32, u32)) -> Self::Cols<F> {
        let (x, y) = input;
        let is_less_than = if x < y { 1 } else { 0 };

        let mut x_bits = vec![];
        for d in 0..self.limb_bits {
            x_bits.push(F::from_canonical_u32((x >> d) & 1));
        }
        let mut y_bits = vec![];
        for d in 0..self.limb_bits {
            y_bits.push(F::from_canonical_u32((y >> d) & 1));
        }
        let mut comparisons = vec![];
        for d in 1..=self.limb_bits {
            let x_prefix = x & ((1 << d) - 1);
            let y_prefix = y & ((1 << d) - 1);
            comparisons.push(F::from_canonical_u32(if x_prefix < y_prefix {
                1
            } else {
                0
            }));
        }

        let io = IsLessThanBitsIOCols {
            x: F::from_canonical_u32(x),
            y: F::from_canonical_u32(y),
            is_less_than: F::from_canonical_u32(is_less_than),
        };
        let aux = IsLessThanBitsAuxCols {
            x_bits,
            y_bits,
            comparisons,
        };

        IsLessThanBitsCols { io, aux }
    }
}
