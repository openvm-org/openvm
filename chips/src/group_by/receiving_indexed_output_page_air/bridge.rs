use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::ReceivingIndexedOutputPageCols, ReceivingIndexedOutputPageAir};
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField64> AirBridge<F> for ReceivingIndexedOutputPageAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            ReceivingIndexedOutputPageCols::<usize>::from_slice(&all_cols, &self.final_air);

        SubAirBridge::sends(&self.final_air, my_final_page_cols.final_page_cols)
    }

    /// Receives GroupBy result (count via is_final, group_by cols (idx) and aggregated cols(data)) from GroupByInput
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            ReceivingIndexedOutputPageCols::<usize>::from_slice(&all_cols, &self.final_air);

        let page_cols = my_final_page_cols.final_page_cols.page_cols;
        let alloc_idx = page_cols.is_alloc;

        let page_cols = page_cols
            .idx
            .iter()
            .copied()
            .chain(page_cols.data)
            .map(VirtualPairCol::single_main)
            .collect::<Vec<_>>();

        let input_count = VirtualPairCol::single_main(alloc_idx);

        vec![Interaction {
            fields: page_cols,
            count: input_count,
            argument_index: self.page_bus_index,
        }]
    }
}
