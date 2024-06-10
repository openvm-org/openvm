use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

// use crate::dummy_hash::DummyHashChip;

// use crate::sub_chip::LocalTraceInstructions;

use super::FlatHashChip;

impl<const N: usize, const R: usize> FlatHashChip<N, R> {
    pub fn generate_trace<F: Field>(&self, x: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let mut state = vec![F::zero(); self.hash_width];
        let mut rows = vec![];
        let num_hashes = self.page_width / self.hash_rate;

        for row in x.iter() {
            let mut new_row = state.clone();
            for hash_index in 0..num_hashes {
                let start = hash_index * self.hash_rate;
                let end = (hash_index + 1) * self.hash_rate;
                let row_slice = &row[start..end];
                state = self.hashchip.request(state.clone(), row_slice.to_vec());
                new_row.extend(state.iter());
            }
            rows.push([row.clone(), new_row].concat());
        }

        RowMajorMatrix::new(rows.concat(), self.get_width())
    }

    // pub fn generate_chips_traces_pis<F: Field>(
    //     &self,
    //     x: Vec<Vec<F>>,
    // ) -> (Vec<RowMajorMatrix<F>>, Vec<Vec<F>>) {
    //     let num_hashes = self.page_width / self.hash_rate;
    //     let mut chips = vec![];
    //     let mut pis = vec![];
    //     for _ in 0..num_hashes {
    //         chips.push(self.clone());
    //         pis.push(vec![F::zero(); self.hash_width]);
    //     }
    // }
}

// impl<F: Field> LocalTraceInstructions<F> for FlatHashChip {
//     type LocalInput = F;

//     fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
//         let is_zero = FlatHashChip::request(local_input);
//         let inv = if is_zero {
//             F::zero()
//         } else {
//             local_input.inverse()
//         };
//         FlatHashCols::<F>::new(local_input, F::from_bool(is_zero), inv)
//     }
// }
