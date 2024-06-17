use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use crate::sub_chip::SubAir;

use super::columns::AssertSortedCols;
use super::AssertSortedAir;

impl<F: Field> BaseAir<F> for AssertSortedAir {
    fn width(&self) -> usize {
        AssertSortedCols::<F>::get_width(
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
            self.is_less_than_tuple_air().tuple_len(),
        )
    }
}

impl<AB: AirBuilder> Air<AB> for AssertSortedAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        // get the current row and the next row
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = AssertSortedCols::from_slice(
            local,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
            self.is_less_than_tuple_air().tuple_len(),
        );

        let next_cols = AssertSortedCols::from_slice(
            next,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
            self.is_less_than_tuple_air().tuple_len(),
        );

        // constrain that the current key is less than the next
        builder
            .when_transition()
            .assert_one(local_cols.less_than_next_key);

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: local_cols.key,
                y: next_cols.key,
                tuple_less_than: local_cols.less_than_next_key,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        // constrain the indicator that we used to check whether the current key < next key is correct
        SubAir::eval(
            self.is_less_than_tuple_air(),
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );
    }
}
