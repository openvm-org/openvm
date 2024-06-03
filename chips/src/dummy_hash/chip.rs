use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::{columns::DummyHashCols, DummyHashChip};

impl<F: Field, const N: usize, const R: usize> Chip<F> for DummyHashChip<F, N, R> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let fields = (0..N).map(|i| VirtualPairCol::single_main(i)).collect();

        let count = vec![VirtualPairCol::constant(F::one()); N];

        vec![Interaction {
            fields,
            count,
            argument_index: self.bus_index(),
        }]
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let fields = (N + R..2 * N + R)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();

        let count = vec![VirtualPairCol::constant(F::one()); N];

        vec![Interaction {
            fields,
            count,
            argument_index: self.bus_index(),
        }]
    }
}
