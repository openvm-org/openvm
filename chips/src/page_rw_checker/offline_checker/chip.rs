use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::sub_chip::SubAirWithInteractions;

// TODO: send the operations on a different bus

impl<F: PrimeField64> SubAirWithInteractions<F> for OfflineChecker {
    fn receives(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let virtual_cols = col_indices
            .page_row
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(col_indices.is_initial),
            argument_index: self.bus_index(),
        }]
    }

    fn sends(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let virtual_cols = col_indices
            .page_row
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(col_indices.is_final),
            argument_index: self.bus_index(),
        }]
    }
}

impl<F: PrimeField64> Chip<F> for OfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = OfflineCheckerCols::<usize>::from_slice(
            &all_cols,
            self.page_width(),
            self.idx_len,
            self.data_len,
        );
        SubAirWithInteractions::receives(self, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = OfflineCheckerCols::<usize>::from_slice(
            &all_cols,
            self.page_width(),
            self.idx_len,
            self.data_len,
        );
        SubAirWithInteractions::sends(self, cols_to_send)
    }
}
