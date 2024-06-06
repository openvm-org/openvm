use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::RANGE_GATE_COL_MAP, RangeCheckerGateAir};

impl<F: PrimeField64> AirBridge<F> for RangeCheckerGateAir {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(RANGE_GATE_COL_MAP.counter)],
            count: VirtualPairCol::single_main(RANGE_GATE_COL_MAP.mult),
            argument_index: self.bus_index,
        }]
    }
}
