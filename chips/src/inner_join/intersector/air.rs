use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_less_than_tuple::columns::IsLessThanTupleIOCols;
use crate::sub_chip::{AirConfig, SubAir};
use crate::utils::or;

use super::columns::{IntersectorAuxCols, IntersectorCols, IntersectorIOCols};
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

        let local_cols = IntersectorCols::from_slice(local_slc, self);
        let next_cols = IntersectorCols::from_slice(next_slc, self);

        SubAir::eval(
            self,
            builder,
            [local_cols.io, next_cols.io],
            [local_cols.aux, next_cols.aux],
        );
    }
}

impl<AB: AirBuilder> SubAir<AB> for IntersectorAir {
    type IoView = [IntersectorIOCols<AB::Var>; 2];
    type AuxView = [IntersectorAuxCols<AB::Var>; 2];

    /// Ensures indices in non-extra rows are sorted and distinct,
    /// that out_mult is correct, and that multiplicity are zero for
    /// non-extra rows
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let (local_io_cols, next_io_cols) = (&io[0], &io[1]);
        let next_aux_cols = &aux[1];

        // Ensuring that rows are sorted by idx
        let lt_io_cols = IsLessThanTupleIOCols {
            x: local_io_cols.idx.clone(),
            y: next_io_cols.idx.clone(),
            tuple_less_than: next_aux_cols.lt_out,
        };

        SubAir::eval(
            &self.lt_chip,
            &mut builder.when_transition(),
            lt_io_cols,
            next_aux_cols.lt_aux.clone(),
        );

        builder.when_transition().assert_one(or::<AB>(
            next_io_cols.is_extra.into(),
            next_aux_cols.lt_out.into(),
        ));

        // Ensuring out_mult is correct
        builder.assert_eq(
            local_io_cols.t1_mult * local_io_cols.t2_mult,
            local_io_cols.out_mult,
        );

        // Ensuting that t1_mult, t2_mult, out_mult are zeros when is_extra is one
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.t1_mult);
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.t2_mult);
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.out_mult);
    }
}
