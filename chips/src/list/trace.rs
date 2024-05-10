use std::mem::transmute;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{ListCols, NUM_LIST_COLS},
    ListChip,
};

impl<const MAX: u32> ListChip<MAX> {
    pub fn generate_trace<F: PrimeField64>(&mut self) -> RowMajorMatrix<F> {
        let mut rows = vec![[F::zero(); NUM_LIST_COLS]; self.vals.len()];

        for (n, row) in rows.iter_mut().enumerate() {
            let cols: &mut ListCols<F> = unsafe { transmute(row) };

            let cur_val = self.vals[n];

            cols.val = F::from_canonical_u32(cur_val);

            if let Some(ref mut range_checker) = self.range_checker {
                let mut range_checker = range_checker.lock().unwrap();
                range_checker.add_count(cur_val);
            }
        }

        RowMajorMatrix::new(rows.concat(), NUM_LIST_COLS)
    }
}
