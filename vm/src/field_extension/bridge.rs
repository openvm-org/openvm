use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use crate::field_extension::columns::FieldExtensionArithmeticIoCols;

use super::FieldExtensionArithmeticAir;

/// Receives all IO columns from another chip on bus 2 (FieldExtensionArithmeticAir::BUS_INDEX).
impl<T: Field> AirBridge<T> for FieldExtensionArithmeticAir {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![Interaction {
            fields: (0..FieldExtensionArithmeticIoCols::<T>::get_width())
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::one(),
            argument_index: Self::BUS_INDEX,
        }]
    }
}
