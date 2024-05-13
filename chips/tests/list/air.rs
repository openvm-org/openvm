use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::columns::NUM_LIST_COLS;
use super::ListChip;

impl<F: Field, const MAX: u32> BaseAir<F> for ListChip<MAX> {
    fn width(&self) -> usize {
        NUM_LIST_COLS
    }
}

impl<AB, const MAX: u32> Air<AB> for ListChip<MAX>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
