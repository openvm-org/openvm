use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::FlatHashAir;
use super::PageController;
use crate::dummy_hash::DummyHashChip;

impl FlatHashAir {
    pub fn generate_trace<F: Field>(
        &self,
        x: Vec<Vec<F>>,
        hash_chip: &mut DummyHashChip<F>,
    ) -> RowMajorMatrix<F> {
        let mut state = vec![F::zero(); self.hash_width];
        let mut rows = vec![];
        let num_hashes = self.page_width / self.hash_rate;

        for row in x.iter() {
            let mut new_row = state.clone();
            for hash_index in 0..num_hashes {
                let start = hash_index * self.hash_rate;
                let end = (hash_index + 1) * self.hash_rate;
                let row_slice = &row[start..end];
                state = hash_chip.request(state.clone(), row_slice.to_vec());
                new_row.extend(state.iter());
            }
            rows.push([row.clone(), new_row].concat());
        }

        RowMajorMatrix::new(rows.concat(), self.get_width())
    }
}

impl<F: Field> PageController<F> {
    pub fn generate_trace(&self, x: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let mut hash_chip = self.hash_chip.lock();
        self.air.generate_trace(x, &mut *hash_chip)
    }
}
