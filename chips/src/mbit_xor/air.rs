use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::NUM_MBIT_XOR_COLS;
use super::MBitXorChip;

impl<F: Field, const M: usize> BaseAir<F> for MBitXorChip<M> {
    fn width(&self) -> usize {
        NUM_MBIT_XOR_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..M).map(F::from_canonical_usize).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB, const M: usize> Air<AB> for MBitXorChip<M>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
