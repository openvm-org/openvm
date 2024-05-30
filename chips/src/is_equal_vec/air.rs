use std::borrow::Borrow;

use super::columns::IsEqualVecCols;
use super::IsEqualVecChip;
use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

// No interactions
impl<F: Field> Chip<F> for IsEqualVecChip {}

impl<F: Field> BaseAir<F> for IsEqualVecChip {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IsEqualVecChip {
    fn eval(&self, builder: &mut AB) {
        let vec_len = self.vec_len();
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsEqualVecCols::<AB::Var>::from_slice(local, vec_len);

        builder.assert_eq(
            (local_cols.prods[0] + local_cols.x[0] - local_cols.y[0]) * local_cols.invs[0],
            AB::F::one(),
        );

        for i in 0..vec_len {
            builder.assert_eq(
                local_cols.prods[i] * local_cols.prods[i],
                local_cols.prods[i],
            );
            builder.assert_eq(
                local_cols.prods[i] * (local_cols.x[i] - local_cols.y[i]),
                AB::F::zero(),
            );
        }

        for i in 0..vec_len - 1 {
            builder.assert_eq(
                local_cols.prods[i] * local_cols.prods[i + 1],
                local_cols.prods[i + 1],
            );

            builder.assert_eq(
                (local_cols.prods[i + 1] - local_cols.prods[i]
                    + AB::F::one()
                    + local_cols.x[i + 1]
                    - local_cols.y[i + 1])
                    * local_cols.invs[i + 1],
                local_cols.prods[i],
            );
        }
    }
}
