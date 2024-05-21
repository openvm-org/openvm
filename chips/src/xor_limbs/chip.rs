use super::columns::XorLimbsCols;
use afs_stark_backend::interaction::{Chip, Interaction};
use p3_field::PrimeField64;

use super::XorLimbsChip;

impl<F: PrimeField64, const N: usize, const M: usize> Chip<F> for XorLimbsChip<N, M> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = XorLimbsCols::<N, M, F>::get_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = XorLimbsCols::<N, M, F>::cols_numbered(&all_cols);
        self.sends_custom(cols_numbered)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = XorLimbsCols::<N, M, F>::get_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = XorLimbsCols::<N, M, F>::cols_numbered(&all_cols);
        self.receives_custom(cols_numbered)
    }
}
