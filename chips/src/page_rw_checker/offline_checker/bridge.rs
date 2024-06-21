use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use crate::sub_chip::SubAirBridge;
use crate::utils::to_vcols;

impl<F: PrimeField> SubAirBridge<F> for OfflineChecker {
    /// Receives page rows (idx, data) for rows tagged with is_initial on page_bus (sent from PageRWAir)
    /// Receives operations (clk, idx, data, op_type) for rows tagged with is_internal on ops_bus
    fn receives(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = to_vcols(&[col_indices.idx.clone(), col_indices.data.clone()].concat());

        let op_cols: Vec<VirtualPairCol<F>> = to_vcols(
            &[
                vec![col_indices.clk],
                col_indices.idx,
                col_indices.data,
                vec![col_indices.op_type],
            ]
            .concat(),
        );

        vec![
            Interaction {
                fields: page_cols,
                count: VirtualPairCol::single_main(col_indices.is_initial),
                argument_index: self.page_bus_index,
            },
            Interaction {
                fields: op_cols,
                count: VirtualPairCol::single_main(col_indices.is_internal),
                argument_index: self.ops_bus_index,
            },
        ]
    }

    /// Sends page rows (idx, data) for rows tagged with is_final on page_bus with multiplicity is_final_x3 (received by MyFinalPageAir)
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = to_vcols(&[col_indices.idx, col_indices.data].concat());

        let mut interactions = vec![Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(col_indices.is_final_write_x3),
            argument_index: self.page_bus_index,
        }];

        interactions.extend(SubAirBridge::sends(
            &self.lt_idx_clk_air,
            IsLessThanTupleCols {
                io: IsLessThanTupleIOCols {
                    x: vec![usize::MAX; 1 + self.idx_len],
                    y: vec![usize::MAX; 1 + self.idx_len],
                    tuple_less_than: usize::MAX,
                },
                aux: col_indices.lt_aux,
            },
        ));

        interactions
    }
}

impl<F: PrimeField> AirBridge<F> for OfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = OfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        SubAirBridge::receives(self, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = OfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        SubAirBridge::sends(self, cols_to_send)
    }
}
