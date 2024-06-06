use crate::sub_chip::SubAirWithInteractions;

use super::{columns::IsLessThanCols, IsLessThanAir};
use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

impl<F: PrimeField64> Chip<F> for IsLessThanAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = IsLessThanCols::<F>::get_width(*self.limb_bits(), *self.decomp());
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = IsLessThanCols::<usize>::from_slice(&all_cols);

        SubAirWithInteractions::sends(self, cols_numbered)
    }
}

impl<F: PrimeField64> SubAirWithInteractions<F> for IsLessThanAir {
    fn sends(&self, col_indices: IsLessThanCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = vec![];

        // we range check the limbs of the lower_bits so that we know each element
        // of lower_bits has at most limb_bits bits
        for i in 0..(*self.num_limbs() + 1) {
            interactions.push(Interaction {
                fields: vec![VirtualPairCol::single_main(col_indices.aux.lower_decomp[i])],
                count: VirtualPairCol::constant(F::one()),
                argument_index: *self.bus_index(),
            });
        }

        interactions
    }
}
