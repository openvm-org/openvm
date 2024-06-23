use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::PageReadAir;
use crate::common::page_cols::PageCols;
use crate::sub_chip::AirConfig;

impl<F: Field> BaseAir<F> for PageReadAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageReadAir {
    type Cols<T> = PageCols<T>;
}

impl<AB: AirBuilder> Air<AB> for PageReadAir {
    fn eval(&self, _builder: &mut AB) {}
}
