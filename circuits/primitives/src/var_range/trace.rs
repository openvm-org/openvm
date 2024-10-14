use std::borrow::BorrowMut;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{VariableRangeCols, NUM_VARIABLE_RANGE_COLS},
    VariableRangeCheckerChip,
};

impl VariableRangeCheckerChip {
    pub fn generate_trace<F: PrimeField>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![F::zero(); self.count.len() * NUM_VARIABLE_RANGE_COLS];
        for (n, row) in rows.chunks_mut(NUM_VARIABLE_RANGE_COLS).enumerate() {
            let cols: &mut VariableRangeCols<F> = row.borrow_mut();
            cols.mult =
                F::from_canonical_u32(self.count[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_VARIABLE_RANGE_COLS)
    }
}
