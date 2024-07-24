use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{dummy_hash::DummyHashChip, FlatHashAir, PageController};

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
            rows.push([vec![F::one()], row.clone(), new_row].concat());
        }

        let mut blank_row = vec![F::zero(); self.get_width()];
        let last_chunk_start = self.get_width() - self.hash_width;
        blank_row[last_chunk_start..last_chunk_start + self.digest_width].copy_from_slice(
            &rows[rows.len() - 1][last_chunk_start..last_chunk_start + self.digest_width],
        );

        let correct_len = x.len().next_power_of_two();
        rows.extend(vec![blank_row.clone(); correct_len - x.len()]);

        RowMajorMatrix::new(rows.concat(), self.get_width())
    }
}

impl<F: Field> PageController<F> {
    pub fn generate_trace(&self, x: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let mut hash_chip = self.hash_chip.lock();
        self.air.generate_trace(x, &mut *hash_chip)
    }
}
