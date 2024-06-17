use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::MyInitialPageAir;
use crate::common::page_cols::PageCols;
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField64> SubAirBridge<F> for MyInitialPageAir {
    /// Sends page rows (idx, data) for every allocated row on page_bus
    /// Some of this is received by OfflineChecker and some by MyFinalPageChip
    fn sends(&self, col_indices: PageCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = col_indices
            .idx
            .into_iter()
            .chain(col_indices.data)
            .map(VirtualPairCol::single_main)
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(col_indices.is_alloc),
            argument_index: self.page_bus,
        }]
    }
}

impl<F: PrimeField64> AirBridge<F> for MyInitialPageAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = PageCols::<usize>::from_slice(&all_cols, self.idx_len, self.data_len);
        SubAirBridge::sends(self, cols_to_send)
    }
}
