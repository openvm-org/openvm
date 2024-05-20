use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::columns::XorCols;
use super::XorBitsChip;

impl<F: Field, const N: usize> BaseAir<F> for XorBitsChip<N> {
    fn width(&self) -> usize {
        XorCols::<N, F>::get_width()
    }
}

impl<AB: AirBuilderWithPublicValues, const N: usize> Air<AB> for XorBitsChip<N>
where
    AB: AirBuilder,
    AB::Var: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let xor_cols = XorCols::<N, AB::Var>::from_slice(local);

        self.impose_constraints(builder, xor_cols);
    }
}
