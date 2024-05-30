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

/// Imposes AIR constaints within each row
/// Indices are as follows:
/// 0 - 2*vec_len-1: vector input
/// 2*vec_len - 3*vec_len-1: cumulative equality AND (answer in index 3*vec_len-1)
/// 3*vec_len - 4*vec_len-1: inverse used to constrain nonzero when equality holds
///
/// At first index naively implements is_equal constraints
/// At every index constrains cumulative NAND difference
/// At every transition index prohibits 0 followed by 1, and constrains
/// 1 with equality must be followed by 1
/// When product does not change, inv is 0, when product changes, inverse is inverse of difference
impl<AB: AirBuilderWithPublicValues> Air<AB> for IsEqualVecChip {
    fn eval(&self, builder: &mut AB) {
        let vec_len = self.vec_len();
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsEqualVecCols::<AB::Var>::from_slice(local, vec_len);

        builder.assert_eq(
            local_cols.prods[0] + (local_cols.x[0] - local_cols.y[0]) * local_cols.invs[0],
            AB::F::one(),
        );

        for i in 0..vec_len {
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
                local_cols.prods[i + 1]
                    + (local_cols.x[i + 1] - local_cols.y[i + 1]) * local_cols.invs[i + 1],
                local_cols.prods[i],
            );
        }
    }
}
