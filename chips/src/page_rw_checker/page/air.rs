use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::{columns::PageCols, PageAir};
use crate::sub_chip::AirConfig;

impl<F: Field> BaseAir<F> for PageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageAir {
    type Cols<T> = PageCols<T>;
}

impl<AB: AirBuilder> Air<AB> for PageAir {
    fn eval(&self, _builder: &mut AB) {}
}
