use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::columns::Poseidon2ChipCols;
use super::Poseidon2Chip;

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl<const WIDTH: usize, T: Field> AirBridge<T> for Poseidon2ChipAir<WIDTH, T> {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![Interaction {
            fields: (1..Poseidon2ChipCols::<T, WIDTH>::NUM_IO_COLS)
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(0),
            argument_index: Self::BUS_INDEX,
        }]
    }

    fn sends(&self) -> Vec<Interaction<T>> {
        vec![Interaction {
            fields: (0..Poseidon2ChipCols::<T>::NUM_IO_COLS)
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(0),
            argument_index: Self::BUS_INDEX,
        }]
    }
}
