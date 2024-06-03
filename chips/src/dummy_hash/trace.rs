use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

use crate::sub_chip::LocalTraceInstructions;

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIOCols},
    DummyHashChip,
};

impl<const N: usize, const R: usize> DummyHashChip<N, R> {
    pub fn generate_trace<F: Field>(
        &self,
        curr_state: Vec<Vec<F>>,
        to_absorb: Vec<Vec<F>>,
    ) -> RowMajorMatrix<F> {
        let rows = curr_state
            .iter()
            .zip(to_absorb.iter())
            .flat_map(|(curr, to_absorb)| {
                let cols = self.generate_trace_row((curr.clone(), to_absorb.clone()));
                cols.flatten()
            })
            .collect::<Vec<_>>();
        RowMajorMatrix::new(rows, DummyHashCols::<F, N, R>::get_width())
    }
}

impl<F: Field, const N: usize, const R: usize> LocalTraceInstructions<F> for DummyHashChip<N, R> {
    type LocalInput = (Vec<F>, Vec<F>);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let (curr_state, to_absorb) = local_input;
        let mut new_state = curr_state.clone();

        for (new, b) in new_state.iter_mut().take(R).zip(to_absorb.iter()) {
            *new += *b;
        }

        DummyHashCols {
            io: DummyHashIOCols {
                curr_state: curr_state.clone(),
                to_absorb: to_absorb.clone(),
                new_state,
            },
            aux: DummyHashAuxCols {},
        }
    }
}
