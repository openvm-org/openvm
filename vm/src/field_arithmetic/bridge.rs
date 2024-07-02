use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::columns::FieldArithmeticCols;
use super::FieldArithmeticAir;

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl<T: Field> AirBridge<T> for FieldArithmeticAir {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![Interaction {
            fields: (1..FieldArithmeticCols::<T>::NUM_IO_COLS)
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(0),
            argument_index: Self::BUS_INDEX,
        }]
    }
}
