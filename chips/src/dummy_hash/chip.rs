use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::{columns::DummyHashCols, DummyHashChip};

impl<F: Field, const N: usize, const R: usize> Chip<F> for DummyHashChip<F, N, R> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(
                DummyHashCols::<F, N, R>.curr_state,
            )],
            count: VirtualPairCol::single_main(DummyHashCols::<F, N, R>.to_absorb),
            argument_index: self.bus_index(),
        }]
    }
}
