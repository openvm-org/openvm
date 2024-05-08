use afs_middleware::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{
    columns::{RANGE_COL_MAP, RANGE_PREPROCESSED_COL_MAP},
    RangeCheckerChip,
};

impl<F: PrimeField64, const MAX: u32> Chip<F> for RangeCheckerChip<MAX> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_preprocessed(
                RANGE_PREPROCESSED_COL_MAP.counter,
            )],
            count: VirtualPairCol::single_main(RANGE_COL_MAP.mult),
            argument_index: self.bus_index(),
        }]
    }
}
