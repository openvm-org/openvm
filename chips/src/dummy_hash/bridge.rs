use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::DummyHashAir;

impl<F: Field> AirBridge<F> for DummyHashAir {
    fn receives(&self) -> Vec<Interaction<F>> {
        let n = self.hash_width;
        let r = self.rate;
        let fields = (0..2 * n + r)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();

        let count = VirtualPairCol::single_main(2 * n + r);

        vec![Interaction {
            fields,
            count,
            argument_index: self.bus_index(),
        }]
    }
}
