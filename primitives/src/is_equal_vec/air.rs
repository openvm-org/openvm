use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIoCols};
use crate::sub_chip::{AirConfig, SubAir};

#[derive(Clone, Copy, Debug)]
pub struct IsEqualVecAir {
    pub vec_len: usize,
}

impl IsEqualVecAir {
    pub fn new(vec_len: usize) -> Self {
        Self { vec_len }
    }

    pub fn request<F: Clone + PartialEq>(&self, x: &[F], y: &[F]) -> bool {
        x == y
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len
    }

    pub fn aux_width(&self) -> usize {
        2 * self.vec_len - 1
    }
}

impl AirConfig for IsEqualVecAir {
    type Cols<T> = IsEqualVecCols<T>;
}

impl<F: Field> BaseAir<F> for IsEqualVecAir {
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
impl<AB: AirBuilder> Air<AB> for IsEqualVecAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let is_equal_vec_cols = IsEqualVecCols::from_slice(local, self.vec_len);

        SubAir::<AB>::eval(self, builder, is_equal_vec_cols.io, is_equal_vec_cols.aux);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsEqualVecAir {
    type IoView = IsEqualVecIoCols<AB::Var>;
    type AuxView = IsEqualVecAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let IsEqualVecIoCols { x, y, is_equal } = io;
        let IsEqualVecAuxCols { prods, invs } = aux;
        let vec_len = self.vec_len;

        // if vec_len == 1, then prods will be empty and we only need to check the constraint that
        // is_equal indicates whether x[0] = y[0]
        if vec_len > 1 {
            // initialize prods[0] = is_equal(x[0], y[0])
            builder.assert_eq(prods[0] + (x[0] - y[0]) * invs[0], AB::F::one());

            for i in 0..vec_len - 1 {
                // constrain prods[i] = 0 if x[i] != y[i]
                builder.assert_zero(prods[i] * (x[i] - y[i]));
            }

            for i in 0..vec_len - 2 {
                // if prod[i] == 0 all after are 0
                builder.assert_eq(prods[i] * prods[i + 1], prods[i + 1]);
                // prods[i] == 1 forces prods[i+1] == is_equal(x[i+1], y[i+1])
                builder.assert_eq(prods[i + 1] + (x[i + 1] - y[i + 1]) * invs[i + 1], prods[i]);
            }

            // if prods[vec_len - 2] == 0, then is_equal == 0
            builder.assert_eq(prods[vec_len - 2] * is_equal, is_equal);
            // if prods[vec_len - 2] == 1, then is_equal == is_equal(x[vec_len - 1], y[vec_len - 1])
            builder.assert_eq(
                is_equal + (x[vec_len - 1] - y[vec_len - 1]) * invs[vec_len - 1],
                prods[vec_len - 2],
            );
        }

        // constrain is_equal = 0 if x[vec_len - 1] != y[vec_len - 1]
        builder.assert_zero(is_equal * (x[vec_len - 1] - y[vec_len - 1]));
    }
}
