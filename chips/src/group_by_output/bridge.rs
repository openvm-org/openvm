use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
};

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField64;

use super::GroupByOutputAir;
use p3_air::VirtualPairCol;

impl<F: PrimeField64> AirBridge<F> for GroupByOutputAir {
    fn receives(&self) -> Vec<Interaction<F>> {
        let fields = (0..self.get_width())
            .map(|i| VirtualPairCol::single_main(i))
            .collect();

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
