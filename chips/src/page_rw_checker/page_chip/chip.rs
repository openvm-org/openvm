use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use crate::sub_chip::SubAirBridge;

use super::columns::PageCols;
use super::PageChip;

impl PageChip {
    fn custom_sends_or_receives<F: PrimeField64>(
        &self,
        col_indices: PageCols<usize>,
    ) -> Vec<Interaction<F>> {
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

impl<F: PrimeField64> SubAirBridge<F> for PageChip {
    fn receives(&self, col_indices: PageCols<usize>) -> Vec<Interaction<F>> {
        if self.is_send {
            return vec![];
        }

        self.custom_sends_or_receives(col_indices)
    }

    fn sends(&self, col_indices: PageCols<usize>) -> Vec<Interaction<F>> {
        if !self.is_send {
            return vec![];
        }

        self.custom_sends_or_receives(col_indices)
    }
}

impl<F: PrimeField64> AirBridge<F> for PageChip {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = PageCols::<F>::cols_numbered(&all_cols, self.idx_len, self.data_len);
        SubAirBridge::receives(self, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = PageCols::<F>::cols_numbered(&all_cols, self.idx_len, self.data_len);
        SubAirBridge::sends(self, cols_to_send)
    }
}
