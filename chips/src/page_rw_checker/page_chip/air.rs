use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::{columns::PageCols, PageChip};
use crate::sub_chip::{AirConfig, SubAir};

impl<F: Field> BaseAir<F> for PageChip {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageChip {
    type Cols<T> = PageCols<T>;
}

impl<AB: AirBuilder> Air<AB> for PageChip {
    fn eval(&self, _builder: &mut AB) {
        if self.is_send {
            // We assume the initial page is properly formatted
        }
    }
}

impl<AB: AirBuilder> SubAir<AB> for PageChip {
    type AuxView = ();
    type IoView = PageCols<AB::Var>;
    fn eval(&self, _builder: &mut AB, _io: Self::IoView, _aux: Self::AuxView) {
        if self.is_send {
            // We assume the initial page is properly formatted
        }
    }
}
