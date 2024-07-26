use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::columns::IsEqualAuxCols;
use super::columns::{IsEqualCols, IsEqualIoCols, NUM_COLS};

#[derive(Default)]
pub struct IsEqualAir;

impl IsEqualAir {
    pub fn request<F: Field>(&self, x: F, y: F) -> bool {
        x == y
    }
}

impl<F: Field> BaseAir<F> for IsEqualAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for IsEqualAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let is_equal_cols: &[AB::Var] = (*local).borrow();

        let is_equal_cols = IsEqualCols::from_slice(is_equal_cols);

        SubAir::<AB>::eval(self, builder, is_equal_cols.io, is_equal_cols.aux);
    }
}

impl AirConfig for IsEqualAir {
    type Cols<T> = IsEqualCols<T>;
}

impl<AB: AirBuilder> SubAir<AB> for IsEqualAir {
    type IoView = IsEqualIoCols<AB::Var>;
    type AuxView = IsEqualAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        builder.assert_eq((io.x - io.y) * aux.inv + io.is_equal, AB::F::one());
        builder.assert_eq((io.x - io.y) * io.is_equal, AB::F::zero());
    }
}
