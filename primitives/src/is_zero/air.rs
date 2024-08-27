use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{IsZeroCols, IsZeroIoCols, NUM_COLS};
use crate::sub_chip::{AirConfig, SubAir};

#[derive(Copy, Clone, Debug, Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroAir;

impl IsZeroAir {
    pub fn request<F: Field>(x: F) -> bool {
        x == F::zero()
    }
}

impl<F: Field> BaseAir<F> for IsZeroAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl AirConfig for IsZeroAir {
    type Cols<T> = IsZeroCols<T>;
}

impl<AB: AirBuilder> Air<AB> for IsZeroAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let is_zero_cols: &IsZeroCols<_> = (*local).borrow();

        SubAir::<AB>::eval(self, builder, is_zero_cols.io, is_zero_cols.inv);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsZeroAir {
    type IoView = IsZeroIoCols<AB::Var>;
    type AuxView = AB::Var;

    fn eval(&self, builder: &mut AB, io: Self::IoView, inv: Self::AuxView) {
        builder.assert_eq(io.x * io.is_zero, AB::F::zero());
        builder.assert_eq(io.is_zero + io.x * inv, AB::F::one());
    }
}

impl IsZeroAir {
    pub fn subair_eval<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: IsZeroIoCols<AB::Expr>,
        inv: AB::Expr,
    ) {
        builder.assert_eq(io.x.clone() * io.is_zero.clone(), AB::F::zero());
        builder.assert_eq(io.is_zero + io.x * inv, AB::F::one());
    }
}
