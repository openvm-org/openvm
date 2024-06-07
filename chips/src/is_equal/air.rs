use std::borrow::Borrow;

use super::columns::{IsEqualAuxCols, IsEqualCols, IsEqualIOCols, NUM_COLS};
use super::IsEqualChip;
use crate::is_zero::columns::IsZeroIOCols;
use crate::is_zero::IsZeroChip;
use crate::sub_chip::{AirConfig, SubAir};
use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

impl<F: Field> BaseAir<F> for IsEqualChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for IsEqualChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let is_equal_cols: &[AB::Var] = (*local).borrow();

        let is_equal_cols = IsEqualCols::from_slice(is_equal_cols);

        SubAir::<AB>::eval(self, builder, is_equal_cols.io, is_equal_cols.aux);
    }
}

impl AirConfig for IsEqualChip {
    type Cols<T> = IsEqualCols<T>;
}

// No interactions
impl<F: Field> Chip<F> for IsEqualChip {}

impl<AB: AirBuilder> SubAir<AB> for IsEqualChip {
    type IoView = IsEqualIOCols<AB::Var>;
    type AuxView = IsEqualAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let is_zero_io = IsZeroIOCols {
            x: io.x - io.y,
            is_zero: io.is_equal.into(),
        };
        SubAir::eval(&IsZeroChip, builder, is_zero_io, aux.inv.into());
    }
}
