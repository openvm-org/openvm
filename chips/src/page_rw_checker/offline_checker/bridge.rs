use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::sub_chip::SubAirBridge;
use crate::utils::to_vcols;

impl<F: PrimeField> AirBridge<F> for OfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = OfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        let general_cols = cols_to_receive.general_cols;

        let mut interactions =
            SubAirBridge::receives(&self.general_offline_checker, general_cols.clone());

        let page_cols = to_vcols(&[general_cols.idx.clone(), general_cols.data.clone()].concat());

        interactions.push(Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(cols_to_receive.is_initial),
            argument_index: self.page_bus_index,
        });

        interactions
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = OfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        let general_cols = cols_to_send.general_cols;

        let mut interactions =
            SubAirBridge::sends(&self.general_offline_checker, general_cols.clone());

        let page_cols = to_vcols(&[general_cols.idx, general_cols.data].concat());

        interactions.push(Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(cols_to_send.is_final_write_x3),
            argument_index: self.page_bus_index,
        });

        interactions
    }
}
