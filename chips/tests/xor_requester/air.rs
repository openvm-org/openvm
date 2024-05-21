use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use super::columns::NUM_XOR_REQUESTER_COLS;
use super::XorRequesterChip;

impl<F: Field, const MAX: usize> BaseAir<F> for XorRequesterChip<MAX> {
    fn width(&self) -> usize {
        NUM_XOR_REQUESTER_COLS
    }
}

impl<AB, const MAX: usize> Air<AB> for XorRequesterChip<MAX>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
