use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::columns::PageOfflineCheckerCols;
use super::PageOfflineChecker;
use crate::sub_chip::SubAirBridge;
use crate::utils::to_vcols;

impl<F: PrimeField> AirBridge<F> for PageOfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = PageOfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        let offline_checker_cols = cols_to_receive.offline_checker_cols;

        let op_cols: Vec<VirtualPairCol<F>> = to_vcols(
            &[
                vec![offline_checker_cols.clk],
                vec![offline_checker_cols.op_type],
                offline_checker_cols.idx.clone(),
                offline_checker_cols.data.clone(),
            ]
            .concat(),
        );

        let page_cols = to_vcols(
            &[
                offline_checker_cols.idx.clone(),
                offline_checker_cols.data.clone(),
            ]
            .concat(),
        );

        vec![
            Interaction {
                fields: page_cols,
                count: VirtualPairCol::single_main(cols_to_receive.is_initial),
                argument_index: self.page_bus_index,
            },
            Interaction {
                fields: op_cols,
                count: VirtualPairCol::single_main(cols_to_receive.is_internal),
                argument_index: self.offline_checker.ops_bus,
            },
        ]
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = PageOfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        let offline_checker_cols = cols_to_send.offline_checker_cols;

        let mut interactions =
            SubAirBridge::sends(&self.offline_checker, offline_checker_cols.clone());

        let page_cols = to_vcols(&[offline_checker_cols.idx, offline_checker_cols.data].concat());

        interactions.push(Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(cols_to_send.is_final_write_x3),
            argument_index: self.page_bus_index,
        });

        interactions
    }
}
