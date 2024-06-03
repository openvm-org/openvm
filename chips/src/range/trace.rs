use std::mem::transmute;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{RangeCols, NUM_RANGE_COLS},
    RangeCheckerChip,
};

impl RangeCheckerChip {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![[F::zero(); NUM_RANGE_COLS]; self.range_max as usize];
        for (n, row) in rows.iter_mut().enumerate() {
            let cols: &mut RangeCols<F> = unsafe { transmute(row) };

            cols.mult =
                F::from_canonical_u32(self.count[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows.concat(), NUM_RANGE_COLS)
    }
}
