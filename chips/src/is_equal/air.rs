use std::borrow::Borrow;

use super::columns::{IsEqualCols, IsEqualIOCols, NUM_COLS};
use super::IsEqualAir;
use crate::sub_chip::{AirConfig, SubAir};
use afs_stark_backend::interaction::AirBridge;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

impl<F: Field> BaseAir<F> for IsEqualAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for IsEqualAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let is_equal_cols: &IsEqualCols<_> = (*local).borrow();

        SubAir::<AB>::eval(self, builder, is_equal_cols.io, is_equal_cols.inv);
    }
}

impl AirConfig for IsEqualAir {
    type Cols<T> = IsEqualCols<T>;
}

// No interactions
impl<F: Field> AirBridge<F> for IsEqualAir {}

impl<AB: AirBuilder> SubAir<AB> for IsEqualAir {
    type IoView = IsEqualIOCols<AB::Var>;
    type AuxView = AB::Var;

    fn eval(&self, builder: &mut AB, io: Self::IoView, inv: Self::AuxView) {
        builder.assert_eq((io.x - io.y) * inv + io.is_equal, AB::F::one());
        builder.assert_eq((io.x - io.y) * io.is_equal, AB::F::zero());
    }
}
