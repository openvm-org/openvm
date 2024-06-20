use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use crate::au::{columns::FieldArithmeticCols, FieldArithmeticAir};

impl<T: Field> AirBridge<T> for FieldArithmeticAir {
    fn receives(&self) -> Vec<Interaction<T>> {
        // (0..AUCols::<T>::NUM_IO_COLS)
        //     .map(|i| Interaction {
        //         fields: vec![VirtualPairCol::single_main(i)],
        //         count: VirtualPairCol::one(),
        //         argument_index: Self::BUS_INDEX,
        //     })
        vec![Interaction {
            fields: (0..FieldArithmeticCols::<T>::NUM_IO_COLS)
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::one(),
            argument_index: Self::BUS_INDEX,
        }]
    }
}
