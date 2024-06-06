use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::LIST_COL_MAP, ListChip};

impl<F: PrimeField32> AirBridge<F> for ListChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(LIST_COL_MAP.val)],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
