use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use crate::au::{columns::AUCols, AUAir};

impl<T: Field> AirBridge<T> for AUAir {
    fn receives(&self) -> Vec<Interaction<T>> {
        (0..AUCols::<T>::NUM_IO_COLS)
            .map(|i| Interaction {
                fields: vec![VirtualPairCol::single_main(i)],
                count: VirtualPairCol::constant(T::one()),
                argument_index: i,
            })
            .collect()
    }
}
