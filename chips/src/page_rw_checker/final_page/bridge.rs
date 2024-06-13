use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::FinalPageCols, FinalPageAir};
use crate::{
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
        IsLessThanTupleAir,
    },
    sub_chip::SubAirBridge,
};

impl<F: PrimeField64> SubAirBridge<F> for FinalPageAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: FinalPageCols<usize>) -> Vec<Interaction<F>> {
        let lt_air = IsLessThanTupleAir::new(
            self.range_bus_index,
            1 << self.idx_decomp,
            vec![self.idx_limb_bits; self.idx_len],
            self.idx_decomp,
        );

        SubAirBridge::sends(
            &lt_air,
            IsLessThanTupleCols {
                io: IsLessThanTupleIOCols {
                    x: vec![usize::MAX; 1 + self.idx_len],
                    y: vec![usize::MAX; 1 + self.idx_len],
                    tuple_less_than: usize::MAX,
                },
                aux: col_indices.aux_cols.lt_cols,
            },
        )
    }

    /// Receives page rows (idx, data) for every allocated row on page_bus
    /// Receives all indices in rows tagged with is_in_ops on checker_final_bus
    fn receives(&self, col_indices: FinalPageCols<usize>) -> Vec<Interaction<F>> {
        let page_cols = col_indices
            .page_cols
            .idx
            .iter()
            .copied()
            .chain(col_indices.page_cols.data)
            .map(VirtualPairCol::single_main)
            .collect::<Vec<_>>();

        let idx_cols = col_indices
            .page_cols
            .idx
            .into_iter()
            .map(VirtualPairCol::single_main)
            .collect();

        vec![
            Interaction {
                fields: page_cols,
                count: VirtualPairCol::single_main(col_indices.page_cols.is_alloc),
                argument_index: self.page_bus_index,
            },
            Interaction {
                fields: idx_cols,
                count: VirtualPairCol::single_main(col_indices.aux_cols.is_in_ops),
                argument_index: self.checker_final_bus_index,
            },
        ]
    }
}

impl<F: PrimeField64> AirBridge<F> for FinalPageAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = FinalPageCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            self.idx_limb_bits,
            self.idx_decomp,
        );

        SubAirBridge::sends(self, cols_to_send)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = FinalPageCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            self.idx_limb_bits,
            self.idx_decomp,
        );

        SubAirBridge::receives(self, cols_to_send)
    }
}
