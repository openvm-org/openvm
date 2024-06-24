use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use crate::fp4_extension::add_sub::{columns::FieldExtensionAddSubCols, FieldExtensionAddSubAir};

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl<T: Field> AirBridge<T> for FieldExtensionAddSubAir {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![Interaction {
            fields: (0..FieldExtensionAddSubCols::<T>::NUM_COLS)
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::one(),
            argument_index: Self::BUS_INDEX,
        }]
    }
}
