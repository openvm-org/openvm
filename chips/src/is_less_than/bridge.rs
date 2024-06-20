use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use crate::sub_chip::SubAirBridge;

use super::{columns::IsLessThanCols, IsLessThanAir};

impl<F: PrimeField> AirBridge<F> for IsLessThanAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = IsLessThanCols::<F>::get_width(self.limb_bits(), self.decomp());
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = IsLessThanCols::<usize>::from_slice(&all_cols);

        SubAirBridge::sends(self, cols_numbered)
    }
}

impl<F: PrimeField> SubAirBridge<F> for IsLessThanAir {
    fn sends(&self, col_indices: IsLessThanCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = vec![];

        // we range check the limbs of the lower_bits so that we know each element
        // of lower_bits has at most limb_bits bits
        for i in 0..(self.num_limbs() + 1) {
            interactions.push(Interaction {
                fields: vec![VirtualPairCol::single_main(col_indices.aux.lower_decomp[i])],
                count: VirtualPairCol::constant(F::one()),
                argument_index: self.bus_index(),
            });
        }

        interactions
    }
}
