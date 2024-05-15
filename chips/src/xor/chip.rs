use afs_middleware::interaction::{Chip, Interaction};
use p3_field::PrimeField64;

use super::{columns::XorCols, XorChip};

impl<F: PrimeField64, const N: usize> Chip<F> for XorChip<N> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = XorCols::<N, F>::get_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = XorCols::<N, F>::cols_to_receive(&all_cols);

        vec![self.receives_custom(cols_to_receive)]
    }
}
