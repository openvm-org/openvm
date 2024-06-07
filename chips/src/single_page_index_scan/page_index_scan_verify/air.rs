use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::AirConfig;

use super::{columns::PageIndexScanVerifyCols, PageIndexScanVerifyAir};

impl AirConfig for PageIndexScanVerifyAir {
    type Cols<T> = PageIndexScanVerifyCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanVerifyAir {
    fn width(&self) -> usize {
        PageIndexScanVerifyCols::<F>::get_width(self.idx_len, self.data_len)
    }
}

impl<AB: AirBuilder> Air<AB> for PageIndexScanVerifyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let _local_cols =
            PageIndexScanVerifyCols::<AB::Var>::from_slice(local, self.idx_len, self.data_len);
    }
}
