use std::iter::zip;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsEqualVecCols, IsEqualVecAir};

impl IsEqualVecAir {
    pub fn generate_trace<F: Field>(&self, x: Vec<Vec<F>>, y: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let height: usize = x.len();
        assert!(height.is_power_of_two());
        assert_eq!(x.len(), y.len());

        let rows: Vec<_> = zip(x, y)
            .flat_map(|(x_row, y_row)| {
                let row = self.generate_trace_row((x_row, y_row));
                row.flatten()
            })
            .collect();

        RowMajorMatrix::new(rows, width)
    }
}

impl<F: Field> LocalTraceInstructions<F> for IsEqualVecAir {
    type LocalInput = (Vec<F>, Vec<F>);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        assert_eq!(self.vec_len, local_input.0.len());
        assert_eq!(self.vec_len, local_input.1.len());
        let (x_row, y_row) = local_input;
        let vec_len = self.vec_len;
        let mut transition_index = 0;
        while transition_index < vec_len && x_row[transition_index] == y_row[transition_index] {
            transition_index += 1;
        }

        let prods = std::iter::repeat(F::one())
            .take(transition_index)
            .chain(std::iter::repeat(F::zero()).take(vec_len - transition_index))
            .collect::<Vec<F>>();

        let is_equal = prods[vec_len - 1];

        let mut invs = std::iter::repeat(F::zero())
            .take(vec_len)
            .collect::<Vec<F>>();

        if transition_index != vec_len {
            invs[transition_index] = (x_row[transition_index] - y_row[transition_index]).inverse();
        }

        IsEqualVecCols::new(x_row, y_row, is_equal, prods[..vec_len - 1].to_vec(), invs)
    }
}
