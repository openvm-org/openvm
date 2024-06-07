use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::PageIndexScanCols, PageIndexScanAir};

impl AirConfig for PageIndexScanAir {
    type Cols<T> = PageIndexScanCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanAir {
    fn width(&self) -> usize {
        PageIndexScanCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air.limb_bits().clone(),
            *self.is_less_than_tuple_air.decomp(),
        )
    }
}

impl<AB: AirBuilder> Air<AB> for PageIndexScanAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = PageIndexScanCols::<AB::Var>::from_slice(
            local,
            self.idx_len,
            self.data_len,
            *self.is_less_than_tuple_air.decomp(),
            self.is_less_than_tuple_air.limb_bits().clone(),
        );

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: local_cols.idx,
                y: local_cols.x,
                tuple_less_than: local_cols.satisfies_pred,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        builder.assert_eq(
            local_cols.is_alloc * local_cols.satisfies_pred,
            local_cols.send_row,
        );
        builder.assert_bool(local_cols.send_row);

        // constrain the indicator that we used to check wheter key < x is correct
        SubAir::eval(
            &self.is_less_than_tuple_air,
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );
    }
}
