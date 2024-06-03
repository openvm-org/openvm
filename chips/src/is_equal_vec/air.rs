use std::borrow::Borrow;

use afs_stark_backend::interaction::Chip;
use p3_air::AirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIOCols},
    IsEqualVecChip,
};

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
/// 0..2*vec_len: vector input
/// 2*vec_len..3*vec_len: cumulative equality AND (answer in index 3*vec_len-1)
/// 3*vec_len..4*vec_len: inverse used to constrain nonzero when equality holds
///
/// At first index naively implements is_equal constraints
/// At every index constrains cumulative NAND difference
/// At every transition index prohibits 0 followed by 1, and constrains
/// 1 with equality must be followed by 1
/// When product does not change, inv is 0, when product changes, inverse is inverse of difference
impl<AB: AirBuilder> Air<AB> for IsEqualVecChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let is_equal_vec_cols = IsEqualVecCols::from_slice(local, self.vec_len);

        SubAir::<AB>::eval(self, builder, is_equal_vec_cols.io, is_equal_vec_cols.aux);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsEqualVecChip {
    type IoView = IsEqualVecIOCols<AB::Var>;
    type AuxView = IsEqualVecAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let IsEqualVecIOCols { x, y, prod: _ } = io;
        let IsEqualVecAuxCols { prods, invs } = aux;
        let vec_len = self.vec_len;
        builder.assert_eq(prods[0] + (x[0] - y[0]) * invs[0], AB::F::one());

        for i in 0..vec_len {
            builder.assert_zero(prods[i] * (x[i] - y[i]));
        }

        for i in 0..vec_len - 1 {
            builder.assert_eq(prods[i] * prods[i + 1], prods[i + 1]);
            builder.assert_eq(prods[i + 1] + (x[i + 1] - y[i + 1]) * invs[i + 1], prods[i]);
        }
    }
}
