use p3_field::{Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IsEqualVecAuxColsMut, IsEqualVecCols},
    IsEqualVecAir,
};
use crate::{is_equal_vec::columns::IsEqualVecColsMut, sub_chip::LocalTraceInstructions};

impl IsEqualVecAir {
    pub fn generate_trace<F: Field>(&self, x: Vec<Vec<F>>, y: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let height: usize = x.len();
        assert!(height.is_power_of_two());
        assert_eq!(x.len(), y.len());

        let mut rows_concat = vec![F::zero(); width * x.len()];
        for (i, (x, y)) in x.iter().zip(y.iter()).enumerate() {
            let mut is_equal_cols =
                IsEqualVecColsMut::from_slice(&mut rows_concat[i * width..(i + 1) * width], self);

            self.generate_trace_row(x, y, &mut is_equal_cols);
        }

        RowMajorMatrix::new(rows_concat, width)
    }

    pub fn generate_trace_row<F: Field>(
        &self,
        x: &[F],
        y: &[F],
        is_equal_cols: &mut IsEqualVecColsMut<F>,
    ) {
        is_equal_cols.io.x.clone_from_slice(x);
        is_equal_cols.io.y.clone_from_slice(y);
        *is_equal_cols.io.is_equal = if x == y { F::one() } else { F::zero() };

        self.generate_trace_row_aux(x, y, &mut is_equal_cols.aux);
    }

    // TODO: should the input always be u32s? This will make more sense when we have the inverse opt in place
    // Assumes that input of IsEqualVecColsMut is filled out
    pub fn generate_trace_row_aux<F: Field>(
        &self,
        x_row: &[F],
        y_row: &[F],
        is_equal_aux_cols: &mut IsEqualVecAuxColsMut<F>,
    ) {
        let vec_len = self.vec_len;
        let mut transition_index = 0;
        while transition_index < vec_len && x_row[transition_index] == y_row[transition_index] {
            transition_index += 1;
        }

        // TODO: test if initializing a mut vec and editting it is more efficient
        let prods: Vec<F> = (0..vec_len - 1)
            .map(|i| {
                if i < transition_index {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();

        let mut invs = vec![F::zero(); vec_len];
        if transition_index != vec_len {
            invs[transition_index] = (x_row[transition_index] - y_row[transition_index]).inverse();
        }

        // Filling out is_equal_cols
        is_equal_aux_cols.prods.clone_from_slice(&prods);
        is_equal_aux_cols.invs.clone_from_slice(&invs);
    }
}

impl<F: PrimeField> LocalTraceInstructions<F> for IsEqualVecAir {
    type LocalInput = (Vec<F>, Vec<F>);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let width = self.get_width();

        let mut row = vec![F::zero(); width];
        let mut is_equal_cols = IsEqualVecColsMut::from_slice(&mut row, self);

        self.generate_trace_row(
            local_input.0.as_slice(),
            local_input.1.as_slice(),
            &mut is_equal_cols,
        );

        IsEqualVecCols::<F>::from_slice(&row, self.vec_len)
    }
}
