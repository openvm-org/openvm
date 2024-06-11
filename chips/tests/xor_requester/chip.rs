use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::XOR_REQUESTER_COL_MAP, XorRequesterChip};

impl<F: PrimeField64, const N: usize> AirBridge<F> for XorRequesterChip<N> {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(XOR_REQUESTER_COL_MAP.x),
                VirtualPairCol::single_main(XOR_REQUESTER_COL_MAP.y),
                VirtualPairCol::single_main(XOR_REQUESTER_COL_MAP.z),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }]
    }
}
