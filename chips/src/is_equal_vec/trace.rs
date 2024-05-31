use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsEqualVecCols, IsEqualVecChip};

impl IsEqualVecChip {
    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let height: usize = self.x.len();
        assert!(height.is_power_of_two());

        let rows: Vec<Vec<F>> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(x_row, y_row)| {
                let row = self.generate_trace_row((x_row.clone(), y_row.clone()));
                row.to_vec()
            })
            .collect();

        RowMajorMatrix::new(rows.concat(), width)
    }
}

impl<F: Field> LocalTraceInstructions<F> for IsEqualVecChip {
    type LocalInput = (Vec<u32>, Vec<u32>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> IsEqualVecCols<F> {
        let (x_row, y_row) = input;
        let vec_len = self.vec_len();
        let mut transition_index = 0;
        while transition_index < vec_len && x_row[transition_index] == y_row[transition_index] {
            transition_index += 1;
        }

        let prods = std::iter::repeat(F::one())
            .take(transition_index)
            .chain(std::iter::repeat(F::zero()).take(vec_len - transition_index))
            .collect::<Vec<F>>();

        let mut invs = std::iter::repeat(F::zero())
            .take(vec_len)
            .collect::<Vec<F>>();

        if transition_index != vec_len {
            invs[transition_index] = (F::from_canonical_u32(x_row[transition_index])
                - F::from_canonical_u32(y_row[transition_index]))
            .inverse();
        }

        IsEqualVecCols::new(
            x_row.iter().map(|x| F::from_canonical_u32(*x)).collect(),
            y_row.iter().map(|y| F::from_canonical_u32(*y)).collect(),
            prods,
            invs,
        )
    }
}
