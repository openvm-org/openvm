use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{columns::NUM_LIST_COLS, ListChip};

impl ListChip {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for val in self.vals.iter() {
            rows.push(vec![F::from_canonical_u32(*val)]);
            self.range_checker.add_count(*val);
        }

        RowMajorMatrix::new(rows.concat(), NUM_LIST_COLS)
    }
}
