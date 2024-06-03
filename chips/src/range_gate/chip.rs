use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::RANGE_GATE_COL_MAP, RangeCheckerGateChip};

impl<F: PrimeField64> Chip<F> for RangeCheckerGateChip {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(RANGE_GATE_COL_MAP.counter)],
            count: VirtualPairCol::single_main(RANGE_GATE_COL_MAP.mult),
            argument_index: self.bus_index(),
        }]
    }
}
