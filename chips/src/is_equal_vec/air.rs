use std::borrow::Borrow;

use crate::sub_chip::SubAir;

use super::columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIOCols};
use super::IsEqualVecChip;
use crate::sub_chip::AirConfig;
use afs_stark_backend::interaction::Chip;
use p3_air::AirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

// No interactions
impl<F: Field> Chip<F> for IsEqualVecChip {}

impl AirConfig for IsEqualVecChip {
    type Cols<T> = IsEqualVecCols<T>;
}

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
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let is_equal_vec_cols = IsEqualVecCols::<AB::Var>::from_slice(local, self.vec_len());

        SubAir::<AB>::eval(self, builder, is_equal_vec_cols.io, is_equal_vec_cols.aux);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsEqualVecChip {
    type IoView = IsEqualVecIOCols<AB::Var>;
    type AuxView = IsEqualVecAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let vec_len = self.vec_len();
        builder.assert_eq(
            aux.prods[0] + (io.x[0] - io.y[0]) * aux.invs[0],
            AB::F::one(),
        );

        for i in 0..vec_len {
            builder.assert_eq(aux.prods[i] * (io.x[i] - io.y[i]), AB::F::zero());
        }

        for i in 0..vec_len - 1 {
            builder.assert_eq(aux.prods[i] * aux.prods[i + 1], aux.prods[i + 1]);
            builder.assert_eq(
                aux.prods[i + 1] + (io.x[i + 1] - io.y[i + 1]) * aux.invs[i + 1],
                aux.prods[i],
            );
        }
    }
}
