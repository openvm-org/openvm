use std::iter;

use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::PageCols;
use super::PageChip;
use crate::sub_chip::SubAirWithInteractions;

impl<F: PrimeField64> SubAirWithInteractions<F> for PageChip {
    fn sends(&self, col_indices: PageCols<usize>) -> Vec<Interaction<F>> {
        let virtual_cols = iter::once(col_indices.is_alloc)
            .chain(col_indices.idx)
            .chain(col_indices.data)
            .map(VirtualPairCol::single_main)
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(col_indices.is_alloc),
            argument_index: self.bus_index(),
        }]
    }
}

impl<F: PrimeField64> Chip<F> for PageChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = PageCols::<usize>::from_slice(&all_cols, self.idx_len, self.data_len);
        SubAirWithInteractions::sends(self, cols_to_send)
    }
}
