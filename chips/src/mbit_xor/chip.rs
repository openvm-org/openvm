use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{
    columns::{MBIT_XOR_COL_MAP, MBIT_XOR_PREPROCESSED_COL_MAP},
    MBitXorChip,
};

impl<F: PrimeField64, const M: usize> Chip<F> for MBitXorChip<M> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(MBIT_XOR_PREPROCESSED_COL_MAP.x),
                VirtualPairCol::single_main(MBIT_XOR_PREPROCESSED_COL_MAP.y),
                VirtualPairCol::single_main(MBIT_XOR_PREPROCESSED_COL_MAP.z),
            ],
            count: VirtualPairCol::single_main(MBIT_XOR_COL_MAP.mult),
            argument_index: self.bus_index(),
        }]
    }
}
