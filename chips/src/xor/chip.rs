use afs_middleware::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::XOR_HELPER_COL_MAP, XorChip};

impl<F: PrimeField64, const N: usize> Chip<F> for XorChip<N> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(XOR_HELPER_COL_MAP.x),
                VirtualPairCol::single_main(XOR_HELPER_COL_MAP.y),
                VirtualPairCol::single_main(XOR_HELPER_COL_MAP.z),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
