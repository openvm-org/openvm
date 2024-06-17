use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use crate::sub_chip::SubAirBridge;

use super::columns::ExecutionCols;
use super::ExecutionAir;

impl<F: PrimeField64> SubAirBridge<F> for ExecutionAir {
    fn sends(&self, col_indices: Self::Cols<usize>) -> Vec<Interaction<F>> {
        let virtual_cols = iter::once(col_indices.clk)
            .chain(col_indices.idx)
            .chain(col_indices.data)
            .chain(iter::once(col_indices.op_type))
            .map(VirtualPairCol::single_main)
            .collect();
        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(col_indices.mult),
            argument_index: self.bus_index,
        }]
    }
}

impl<F: PrimeField64> AirBridge<F> for ExecutionAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered =
            ExecutionCols::<usize>::from_slice(&all_cols, self.idx_len, self.data_len);
        SubAirBridge::sends(self, cols_numbered)
    }
}
