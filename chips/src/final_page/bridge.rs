use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField;

use super::{columns::FinalPageCols, FinalPageAir};
use crate::{
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
        IsLessThanTupleAir,
    },
    sub_chip::SubAirBridge,
};

impl<F: PrimeField> SubAirBridge<F> for FinalPageAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    fn sends(&self, col_indices: FinalPageCols<usize>) -> Vec<Interaction<F>> {
        let lt_air = IsLessThanTupleAir::new(
            self.range_bus_index,
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
}

impl<F: PrimeField> AirBridge<F> for FinalPageAir {
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
}
