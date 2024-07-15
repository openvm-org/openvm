use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIOCols},
    DummyHashAir, DummyHashChip,
};

impl<F: Field> DummyHashChip<F> {
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let rows = (0..self.hash_in_states.len())
            .flat_map(|i| {
                let mut combined = vec![F::one()];
                combined.extend(self.hash_in_states[i].clone());
                combined.extend(self.hash_slices[i].clone());
                combined.extend(self.hash_out_states[i].clone());
                combined.into_iter()
            })
            .collect::<Vec<_>>();
        let mut padded_rows = rows.clone();
        let current_len = self.hash_in_states.len();
        let next_power_of_two = current_len.next_power_of_two();
        if next_power_of_two > current_len {
            let width = self.air.get_width();
            let zero_row = vec![F::zero(); width * (next_power_of_two - current_len)];
            padded_rows.extend(zero_row);
        }
        RowMajorMatrix::new(padded_rows, self.air.get_width())
    }
}

impl DummyHashAir {
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
        RowMajorMatrix::new(rows, self.get_width())
    }
}

impl<F: Field> LocalTraceInstructions<F> for DummyHashAir {
    type LocalInput = (Vec<F>, Vec<F>);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let (curr_state, to_absorb) = local_input;
        let new_state = DummyHashAir::hash(curr_state.clone(), to_absorb.clone());

        DummyHashCols {
            io: DummyHashIOCols {
                is_alloc: F::one(),
                curr_state: curr_state.clone(),
                to_absorb: to_absorb.clone(),
                new_state,
            },
            aux: DummyHashAuxCols {},
            width: self.hash_width,
            rate: self.rate,
        }
    }
}
