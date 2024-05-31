use std::borrow::Borrow;

use super::columns::{IsZeroCols, IsZeroIOCols, NUM_COLS};
use super::IsZeroChip;
use crate::sub_chip::{AirConfig, SubAir};
use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

impl<F: Field> BaseAir<F> for IsZeroChip<F> {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> AirConfig for IsZeroChip<F> {
    type Cols<T> = IsZeroCols<T>;
}

// No interactions
impl<F: Field> Chip<F> for IsZeroChip<F> {}

impl<F: Field, AB: AirBuilderWithPublicValues<F = F>> Air<AB> for IsZeroChip<F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let is_zero_cols: &IsZeroCols<_> = (*local).borrow();

        SubAir::<AB>::eval(self, builder, is_zero_cols.io, is_zero_cols.inv);
    }
}

impl<F: Field, AB: AirBuilder> SubAir<AB> for IsZeroChip<F> {
    type IoView = IsZeroIOCols<AB::Var>;
    type AuxView = AB::Var;

    fn eval(&self, builder: &mut AB, io: Self::IoView, inv: Self::AuxView) {
        builder.assert_eq(io.x * io.is_zero, AB::F::zero());
        builder.assert_eq(io.is_zero + io.x * inv, AB::F::one());
    }
}
