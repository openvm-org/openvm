use std::mem::transmute;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{RangeCols, NUM_RANGE_COLS},
    RangeCheckerChip,
};

impl<const MAX: u32> RangeCheckerChip<MAX> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![[F::zero(); NUM_RANGE_COLS]; MAX as usize];
        for (n, row) in rows.iter_mut().enumerate() {
            let cols: &mut RangeCols<F> = unsafe { transmute(row) };
            // FIXME: This is very inefficient when the range is large.
            // Iterate over key/val pairs instead in a separate loop.
            if let Some(c) = self.count.get(&(n as u32)) {
                cols.mult = F::from_canonical_u32(*c);
            }
        }
        RowMajorMatrix::new(rows.concat(), NUM_RANGE_COLS)
    }
}
