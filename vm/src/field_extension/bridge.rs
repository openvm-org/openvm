use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::{PairCol, VirtualPairCol};
use p3_field::Field;

use crate::cpu::{FIELD_EXTENSION_BUS, MEMORY_BUS, WORD_SIZE};

use super::{columns::FieldExtensionArithmeticCols, FieldExtensionArithmeticAir};

fn get_rw_interactions<T: Field>(
    is_write: bool,
    cols_numbered: &FieldExtensionArithmeticCols<usize>,
    addr_space: usize,
    address: usize,
    ext_element_ind: usize,
) -> Vec<Interaction<T>> {
    let mut interactions = vec![];

    let ext_element = if ext_element_ind == 0 {
        cols_numbered.io.x
    } else if ext_element_ind == 1 {
        cols_numbered.io.y
    } else {
        cols_numbered.io.z
    };

    for (i, &element) in ext_element.iter().enumerate() {
        let timestamp = VirtualPairCol::new(
            vec![(PairCol::Main(cols_numbered.aux.start_timestamp), T::one())],
            T::from_canonical_usize(ext_element_ind * 4 + i),
        );

        let pointer = VirtualPairCol::new(
            vec![(PairCol::Main(address), T::from_canonical_usize(1))],
            T::from_canonical_usize(i * WORD_SIZE),
        );

        let mut fields = vec![
            timestamp,
            VirtualPairCol::constant(T::from_bool(is_write)),
            VirtualPairCol::single_main(addr_space),
            pointer,
        ];

        // handle WORD_SIZE > 1 later
        fields.push(VirtualPairCol::single_main(element));

        interactions.push(Interaction {
            fields,
            count: VirtualPairCol::one(),
            argument_index: MEMORY_BUS,
        });
    }

    interactions
}

/// Receives all IO columns from another chip on bus 4 (FieldExtensionArithmeticAir::BUS_INDEX).
impl<T: Field> AirBridge<T> for FieldExtensionArithmeticAir {
    fn sends(&self) -> Vec<Interaction<T>> {
        let all_cols = (0..FieldExtensionArithmeticCols::<T>::get_width()).collect::<Vec<usize>>();
        let cols_numbered = FieldExtensionArithmeticCols::<usize>::from_slice(&all_cols);

        let mut interactions = vec![];

        // reads for x
        interactions.extend(get_rw_interactions(
            false,
            &cols_numbered,
            cols_numbered.aux.d,
            cols_numbered.aux.op_b,
            0,
        ));
        // reads for y
        interactions.extend(get_rw_interactions(
            false,
            &cols_numbered,
            cols_numbered.aux.e,
            cols_numbered.aux.op_c,
            1,
        ));
        // writes for z
        interactions.extend(get_rw_interactions(
            true,
            &cols_numbered,
            cols_numbered.aux.d,
            cols_numbered.aux.op_a,
            2,
        ));

        interactions
    }

    fn receives(&self) -> Vec<Interaction<T>> {
        let all_cols = (0..FieldExtensionArithmeticCols::<T>::get_width()).collect::<Vec<usize>>();
        let cols_numbered = FieldExtensionArithmeticCols::<usize>::from_slice(&all_cols);

        vec![Interaction {
            fields: vec![
                VirtualPairCol::single_main(cols_numbered.io.opcode),
                VirtualPairCol::single_main(cols_numbered.aux.op_a),
                VirtualPairCol::single_main(cols_numbered.aux.op_b),
                VirtualPairCol::single_main(cols_numbered.aux.op_c),
                VirtualPairCol::single_main(cols_numbered.aux.d),
                VirtualPairCol::single_main(cols_numbered.aux.e),
            ],
            count: VirtualPairCol::one(),
            argument_index: FIELD_EXTENSION_BUS,
        }]
    }
}
