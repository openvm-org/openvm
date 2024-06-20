use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::{columns::MyFinalPageCols, MyFinalPageAir};
use crate::{sub_chip::SubAirBridge, utils::to_vcols};

impl<F: PrimeField> AirBridge<F> for MyFinalPageAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols = MyFinalPageCols::<usize>::from_slice(&all_cols, &self.final_air);

        SubAirBridge::sends(&self.final_air, my_final_page_cols.final_page_cols)
    }

    /// Receives page rows (idx, data) for all rows with multiplicity rcv_mult on page_bus
    /// Some of this is sent by PageRWAir and some by OfflineChecker
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols = MyFinalPageCols::<usize>::from_slice(&all_cols, &self.final_air);

        let page_cols = my_final_page_cols.final_page_cols.page_cols;
        let rcv_mult = my_final_page_cols.rcv_mult;

        let page_cols = to_vcols(&[page_cols.idx, page_cols.data].concat());

        vec![Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(rcv_mult),
            argument_index: self.page_bus_index,
        }]
    }
}
