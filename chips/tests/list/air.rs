use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::columns::NUM_LIST_COLS;
use super::ListChip;

impl<F: Field> BaseAir<F> for ListChip {
    fn width(&self) -> usize {
        NUM_LIST_COLS
    }
}

impl<AB> Air<AB> for ListChip
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
