use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use crate::cpu::{MEMORY_BUS, RANGE_CHECKER_BUS};

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use afs_chips::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use afs_chips::is_less_than_tuple::IsLessThanTupleAir;
use afs_chips::sub_chip::SubAirBridge;

impl<const WORD_SIZE: usize, F: PrimeField64> SubAirBridge<F> for OfflineChecker<WORD_SIZE> {
    /// Receives operations (clk, op_type, addr_space, pointer, data)
    fn receives(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let op_cols: Vec<VirtualPairCol<F>> = iter::once(col_indices.clk)
            .chain(iter::once(col_indices.op_type))
            .chain(col_indices.mem_row.iter().copied())
            .map(VirtualPairCol::single_main)
            .collect();

        vec![Interaction {
            fields: op_cols,
            count: VirtualPairCol::single_main(col_indices.is_valid),
            argument_index: MEMORY_BUS,
        }]
    }

    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: OfflineCheckerCols<usize>) -> Vec<Interaction<F>> {
        let lt_air = IsLessThanTupleAir::new(
            RANGE_CHECKER_BUS,
            self.addr_clk_limb_bits.clone(),
            self.decomp,
        );

        SubAirBridge::sends(
            &lt_air,
            IsLessThanTupleCols {
                io: IsLessThanTupleIOCols {
                    x: vec![usize::MAX; 3],
                    y: vec![usize::MAX; 3],
                    tuple_less_than: usize::MAX,
                },
                aux: col_indices.lt_aux,
            },
        )
    }
}

impl<const WORD_SIZE: usize, F: PrimeField64> AirBridge<F> for OfflineChecker<WORD_SIZE> {
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
