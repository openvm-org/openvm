use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::{columns::IndexedPageWriteCols, IndexedPageWriteAir};
use crate::{sub_chip::SubAirBridge, utils::to_vcols};

impl<F: PrimeField> AirBridge<F> for IndexedPageWriteAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            IndexedPageWriteCols::<usize>::from_slice(&all_cols, &self.final_air);

        SubAirBridge::sends(&self.final_air, my_final_page_cols.final_page_cols)
    }

    /// Receives page rows (idx, data) for all rows with multiplicity rcv_mult on page_bus
    /// Some of this is sent by PageRWAir and some by OfflineChecker
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            IndexedPageWriteCols::<usize>::from_slice(&all_cols, &self.final_air);
        SubAirBridge::receives(self, my_final_page_cols)
    }
}

impl<F: PrimeField> SubAirBridge<F> for IndexedPageWriteAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: IndexedPageWriteCols<usize>) -> Vec<Interaction<F>> {
        SubAirBridge::sends(&self.final_air, col_indices.final_page_cols)
    }

    /// Receives page rows (idx, data) for all rows with multiplicity rcv_mult on page_bus
    /// Some of this is sent by PageRWAir and some by OfflineChecker
    fn receives(&self, col_indices: IndexedPageWriteCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = col_indices.final_page_cols.page_cols;
        let rcv_mult = col_indices.rcv_mult;
        let page_cols = to_vcols(&[page_cols.idx, page_cols.data].concat());

        vec![Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(rcv_mult),
            argument_index: self.page_bus_index,
        }]
    }
}
