use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use crate::{
    is_less_than::columns::{IsLessThanCols, IsLessThanIOCols},
    sub_chip::SubAirBridge,
};

use super::{columns::SumCols, SumAir};

impl<F: PrimeField64> AirBridge<F> for SumAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let cols = SumCols::<F>::index_map(self.is_lt_air.limb_bits(), self.is_lt_air.decomp());

        let is_lt_cols = IsLessThanCols {
            // io is unused in the IsLessThan bridge
            io: IsLessThanIOCols {
                x: usize::MAX,
                y: usize::MAX,
                less_than: usize::MAX,
            },
            aux: cols.is_lt_aux_cols,
        };
        let subchip_interactions = SubAirBridge::<F>::sends(&self.is_lt_air, is_lt_cols);

        let mut interactions = vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(cols.key),
                VirtualPairCol::single_main(cols.partial_sum),
            ],
            count: VirtualPairCol::single_main(cols.is_final),
            argument_index: self.output_bus,
        }];
        interactions.extend(subchip_interactions);
        interactions
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let cols = SumCols::<F>::index_map(self.is_lt_air.limb_bits(), self.is_lt_air.decomp());
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(cols.key),
                VirtualPairCol::single_main(cols.value),
            ],
            count: VirtualPairCol::one(),
            argument_index: self.input_bus,
        }]
    }
}
