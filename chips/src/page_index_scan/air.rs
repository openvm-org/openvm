use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAir,
};

use super::{columns::PageIndexScanCols, PageIndexScanAir};

impl<F: Field> BaseAir<F> for PageIndexScanAir {
    fn width(&self) -> usize {
        PageIndexScanCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.limb_bits.clone(),
            self.decomp,
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
            self.decomp,
            self.limb_bits.clone(),
        );

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: local_cols.idx,
                y: local_cols.x,
                tuple_less_than: local_cols.satisfies_pred,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        // constrain the indicator that we used to check wheter key < x is correct
        SubAir::eval(
            &self.is_less_than_tuple_air,
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );
    }
}
