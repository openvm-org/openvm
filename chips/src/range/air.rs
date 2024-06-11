use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::NUM_RANGE_COLS;
use super::RangeCheckerAir;

impl<F: Field> BaseAir<F> for RangeCheckerAir {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..self.range_max).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB> Air<AB> for RangeCheckerAir
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
