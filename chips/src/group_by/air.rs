use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_equal_vec::columns::{IsEqualVecCols, IsEqualVecIOCols};
use crate::sub_chip::SubAir;

use super::columns::GroupByCols;
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

        let is_equal_vec_cols = IsEqualVecCols {
            io: IsEqualVecIOCols {
                x: local_cols.sorted_group_by,
                y: next_cols.sorted_group_by,
                prod: local_cols.eq_next,
            },
            aux: local_cols.is_equal_vec_aux,
        };

        // constrain eq_next to hold the correct value
        SubAir::eval(
            &self.is_equal_vec_air,
            &mut builder.when_transition(),
            is_equal_vec_cols.io,
            is_equal_vec_cols.aux,
        );
        builder.when_last_row().assert_zero(local_cols.eq_next);

        builder.assert_one(local_cols.eq_next + local_cols.is_final);

        builder.when_transition().assert_eq(
            next_cols.partial_aggregated,
            local_cols.eq_next * local_cols.partial_aggregated + local_cols.aggregated,
        );
    }
}
