use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
};

use super::columns::AssertSortedCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField64;

use super::AssertSortedAir;

impl<F: PrimeField64> AirBridge<F> for AssertSortedAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = AssertSortedCols::<F>::get_width(
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = AssertSortedCols::<usize>::from_slice(
            &all_cols,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );

        // range check the decompositions of x within aux columns; here the io doesn't matter
        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: cols_numbered.key.clone(),
                y: cols_numbered.key.clone(),
                tuple_less_than: cols_numbered.less_than_next_key,
            },
            aux: cols_numbered.is_less_than_tuple_aux,
        };

        let subchip_interactions =
            SubAirBridge::<F>::sends(self.is_less_than_tuple_air(), is_less_than_tuple_cols);

        subchip_interactions
    }
}
