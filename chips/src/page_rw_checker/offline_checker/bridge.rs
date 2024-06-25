use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField64> SubAirBridge<F> for OfflineChecker {
    /// Receives page rows (idx, data) for rows tagged with is_initial on page_bus (sent from PageRWAir)
    /// Receives operations (clk, idx, data, op_type) for rows tagged with is_internal on ops_bus
    fn receives(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = col_indices.page_row[1..]
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        let op_cols: Vec<VirtualPairCol<F>> = iter::once(col_indices.clk)
            .chain(col_indices.page_row[1..].iter().copied())
            .chain(iter::once(col_indices.op_type))
            .map(VirtualPairCol::single_main)
            .collect();

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
        let page_cols = col_indices.page_row[1..]
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        let mut interactions = vec![Interaction {
            fields: page_cols,
            count: VirtualPairCol::single_main(col_indices.is_final_x3),
            argument_index: self.page_bus_index,
        }];

        let lt_air = IsLessThanTupleAir::new(
            self.range_bus_index,
            self.idx_clk_limb_bits.clone(),
            self.idx_decomp,
        );

        interactions.extend(SubAirBridge::sends(
            &lt_air,
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

impl<F: PrimeField64> AirBridge<F> for OfflineChecker {
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
