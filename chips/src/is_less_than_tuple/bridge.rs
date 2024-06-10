use crate::{
    is_less_than::columns::{IsLessThanAuxCols, IsLessThanCols, IsLessThanIOCols},
    sub_chip::SubAirBridge,
};
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField64;

use super::{columns::IsLessThanTupleCols, IsLessThanTupleAir};

impl<F: PrimeField64> AirBridge<F> for IsLessThanTupleAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = IsLessThanTupleCols::<F>::get_width(
            self.limb_bits().clone(),
            self.decomp(),
            self.tuple_len(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = IsLessThanTupleCols::<usize>::from_slice(
            &all_cols,
            self.limb_bits().clone(),
            self.decomp(),
            self.tuple_len(),
        );

        SubAirBridge::sends(self, cols_numbered)
    }
}

impl<F: PrimeField64> SubAirBridge<F> for IsLessThanTupleAir {
    fn sends(&self, col_indices: IsLessThanTupleCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = vec![];

        // we need to get the interactions from the IsLessThan subchip
        for i in 0..self.tuple_len() {
            let is_less_than_cols = IsLessThanCols {
                io: IsLessThanIOCols {
                    x: col_indices.io.x[i],
                    y: col_indices.io.y[i],
                    less_than: col_indices.aux.less_than[i],
                },
                aux: IsLessThanAuxCols {
                    lower: col_indices.aux.less_than_aux[i].lower,
                    lower_decomp: col_indices.aux.less_than_aux[i].lower_decomp.clone(),
                },
            };

            let curr_interactions =
                SubAirBridge::<F>::sends(&self.is_less_than_airs[i], is_less_than_cols);
            interactions.extend(curr_interactions);
        }

        interactions
    }
}
