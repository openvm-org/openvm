use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::columns::GeneralOfflineCheckerCols;
use super::GeneralOfflineChecker;
use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField> SubAirBridge<F> for GeneralOfflineChecker {
    /// Receives operations (clk, op_type, addr_space, pointer, data)
    fn receives(&self, col_indices: GeneralOfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let op_cols: Vec<VirtualPairCol<F>> = iter::once(col_indices.clk)
            .chain(iter::once(col_indices.op_type))
            .chain(col_indices.idx.iter().copied())
            .chain(col_indices.data.iter().copied())
            .map(VirtualPairCol::single_main)
            .collect();

        vec![Interaction {
            fields: op_cols,
            count: VirtualPairCol::single_main(col_indices.is_valid),
            argument_index: self.ops_bus,
        }]
    }

    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: GeneralOfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let lt_air =
            IsLessThanTupleAir::new(self.range_bus, self.idx_clk_limb_bits.clone(), self.decomp);

        SubAirBridge::sends(
            &lt_air,
            IsLessThanTupleCols {
                io: IsLessThanTupleIOCols {
                    x: vec![usize::MAX; self.idx_len + 1],
                    y: vec![usize::MAX; self.idx_len + 1],
                    tuple_less_than: usize::MAX,
                },
                aux: col_indices.lt_aux,
            },
        )
    }
}

impl<F: PrimeField> AirBridge<F> for GeneralOfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = GeneralOfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        SubAirBridge::receives(self, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = GeneralOfflineCheckerCols::<usize>::from_slice(&all_cols, self);
        SubAirBridge::sends(self, cols_to_send)
    }
}
