use std::iter;

use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::PageReadCols;
use super::PageReadChip;

impl PageReadChip {
    // receives: ([index] | [page]) mult times
    pub fn receives_custom<F: PrimeField64>(
        &self,
        cols: PageReadCols<usize>,
    ) -> Vec<Interaction<F>> {
        let virtual_cols = iter::once(cols.index)
            .chain(cols.page_row)
            .map(VirtualPairCol::single_main)
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols.mult),
            argument_index: self.bus_index(),
        }]
    }
}

impl<F: PrimeField64> Chip<F> for PageReadChip {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageReadCols::<F>::cols_numbered(&all_cols);
        self.receives_custom(cols_numbered)
    }
}
