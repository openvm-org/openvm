use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::columns::IntersectorCols;
use super::IntersectorAir;

impl<F: Field> BaseAir<F> for IntersectorAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for IntersectorAir {
    type Cols<T> = IntersectorCols<T>;
}

impl<AB: AirBuilder> Air<AB> for IntersectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let (local_slc, next_slc) = ((*local).borrow(), (*next).borrow());

        let local_cols = IntersectorCols::from_slice(local_slc, self.idx_len);
        let next_cols = IntersectorCols::from_slice(next_slc, self.idx_len);

        builder.assert_eq(local_cols.t1_mult * local_cols.t2_mult, local_cols.out_mult);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IntersectorAir {
    type IoView = [IntersectorCols<AB::Var>; 2];
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let (local_cols, next_cols) = (&io[0], &io[1]);

        // TODO: make sure the rows are sorted

        // Ensuting out_mult is correct
        builder.assert_eq(local_cols.t1_mult * local_cols.t2_mult, local_cols.out_mult);

        // Ensuting that t1_mult, t2_mult, out_mult are zeros when is_extra is one
        builder.assert_zero(local_cols.is_extra * local_cols.t1_mult);
        builder.assert_zero(local_cols.is_extra * local_cols.t2_mult);
        builder.assert_zero(local_cols.is_extra * local_cols.out_mult);
    }
}
