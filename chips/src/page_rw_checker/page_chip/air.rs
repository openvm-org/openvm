use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::{columns::PageCols, PageChip};
use crate::sub_chip::AirConfig;

impl<F: Field> BaseAir<F> for PageChip {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageChip {
    type Cols<T> = PageCols<T>;
}

impl<AB: AirBuilder> Air<AB> for PageChip {
    fn eval(&self, _builder: &mut AB) {}
}
