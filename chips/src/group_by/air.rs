use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_equal_vec::columns::{IsEqualVecCols, IsEqualVecIOCols};
use crate::sub_chip::{AirConfig, SubAir};

use super::columns::{GroupByAuxCols, GroupByCols, GroupByIOCols};
use super::GroupByAir;

impl<F: Field> BaseAir<F> for GroupByAir {
    fn width(&self) -> usize {
        GroupByCols::<F>::get_width(self)
    }
}

impl<AB: AirBuilder> Air<AB> for GroupByAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        // get the current row and the next row
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = GroupByCols::<AB::Var>::from_slice(local, self);

        let next_cols = GroupByCols::<AB::Var>::from_slice(next, self);

        SubAir::eval(
            self,
            builder,
            (local_cols.io, next_cols.io),
            (local_cols.aux, next_cols.aux),
        );
    }
}

impl AirConfig for GroupByAir {
    type Cols<T> = GroupByCols<T>;
}

impl<AB: AirBuilder> SubAir<AB> for GroupByAir {
    // io.0 is local.io, io.1 is next.io
    type IoView = (GroupByIOCols<AB::Var>, GroupByIOCols<AB::Var>);
    // aux.0 is local.aux, aux.1 is next.aux
    type AuxView = (GroupByAuxCols<AB::Var>, GroupByAuxCols<AB::Var>);

    fn eval(&self, builder: &mut AB, _io: Self::IoView, aux: Self::AuxView) {
        let is_equal_vec_cols = IsEqualVecCols {
            io: IsEqualVecIOCols {
                x: aux.0.sorted_group_by,
                y: aux.1.sorted_group_by,
                prod: aux.0.eq_next,
            },
            aux: aux.0.is_equal_vec_aux,
        };

        // constrain eq_next to hold the correct value
        SubAir::eval(
            &self.is_equal_vec_air,
            &mut builder.when_transition(),
            is_equal_vec_cols.io,
            is_equal_vec_cols.aux,
        );
        builder.when_last_row().assert_zero(aux.0.eq_next);

        builder.assert_one(aux.0.eq_next + aux.0.is_final);

        builder.when_transition().assert_eq(
            aux.1.aggregated,
            aux.0.eq_next * aux.0.aggregated + aux.0.aggregated,
        );
    }
}
