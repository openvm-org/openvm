use afs_stark_backend::interaction::{Chip, Interaction};
use p3_field::PrimeField64;

use super::columns::PageReadCols;
use super::PageReadChip;

impl<F: PrimeField64> Chip<F> for PageReadChip {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.get_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageReadCols::<F>::cols_numbered(&all_cols);
        self.receives_custom(cols_numbered)
    }
}
