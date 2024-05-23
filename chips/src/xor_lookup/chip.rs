use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{
    columns::{XOR_LOOKUP_COL_MAP, XOR_LOOKUP_PREPROCESSED_COL_MAP},
    XorLookupChip,
};

impl<F: PrimeField64, const M: usize> Chip<F> for XorLookupChip<M> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_preprocessed(XOR_LOOKUP_PREPROCESSED_COL_MAP.x),
                VirtualPairCol::single_preprocessed(XOR_LOOKUP_PREPROCESSED_COL_MAP.y),
                VirtualPairCol::single_preprocessed(XOR_LOOKUP_PREPROCESSED_COL_MAP.z),
            ],
            count: VirtualPairCol::single_main(XOR_LOOKUP_COL_MAP.mult),
            argument_index: self.bus_index(),
        }]
    }
}
