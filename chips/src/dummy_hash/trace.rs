use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

use crate::sub_chip::LocalTraceInstructions;

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIOCols},
    DummyHashChip,
};

impl<const N: usize, const R: usize> DummyHashChip<N, R> {
    pub fn generate_trace<F: Field>(&self, _x: Vec<F>) -> RowMajorMatrix<F> {
        let rows = vec![];
        RowMajorMatrix::new(rows, DummyHashCols::<F, N, R>::get_width())
    }
}

impl<F: Field, const N: usize, const R: usize> LocalTraceInstructions<F> for DummyHashChip<N, R> {
    type LocalInput = ([F; N], [F; R]);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        DummyHashCols {
            io: DummyHashIOCols {
                curr_state: local_input.0,
                to_absorb: local_input.1,
            },
            aux: DummyHashAuxCols {},
        }
    }
}
