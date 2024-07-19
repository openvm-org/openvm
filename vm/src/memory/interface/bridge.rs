use p3_air::{PairCol, VirtualPairCol};
use p3_field::Field;

use afs_stark_backend::interaction::{AirBridge, Interaction};

use crate::memory::interface::air::MemoryInterfaceAir;
use crate::memory::interface::columns::MemoryInterfaceCols;
use crate::memory::interface::{EXPAND_BUS, MEMORY_INTERFACE_BUS};

impl<const CHUNK: usize, F: Field> AirBridge<F> for MemoryInterfaceAir<CHUNK> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let all_cols = (0..MemoryInterfaceCols::<CHUNK, F>::get_width()).collect::<Vec<usize>>();
        let cols_numbered = MemoryInterfaceCols::<CHUNK, usize>::from_slice(&all_cols);

        let mut expand_fields = vec![
            VirtualPairCol::new_main(
                vec![(cols_numbered.direction, F::two().inverse().neg())],
                F::two().inverse(),
            ),
            VirtualPairCol::single_main(cols_numbered.address_space),
            VirtualPairCol::constant(F::zero()),
            VirtualPairCol::single_main(cols_numbered.leaf_label),
        ];
        expand_fields.extend(cols_numbered.values.map(VirtualPairCol::single_main));

        vec![Interaction {
            fields: expand_fields,
            count: VirtualPairCol::single_main(cols_numbered.direction),
            argument_index: EXPAND_BUS,
        }]
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let all_cols = (0..MemoryInterfaceCols::<CHUNK, F>::get_width()).collect::<Vec<usize>>();
        let cols_numbered = MemoryInterfaceCols::<CHUNK, usize>::from_slice(&all_cols);

        (0..CHUNK)
            .map(|i| Interaction {
                fields: vec![
                    VirtualPairCol::single_main(cols_numbered.temp_is_final[i]),
                    VirtualPairCol::single_main(cols_numbered.address_space),
                    VirtualPairCol::new(
                        vec![(
                            PairCol::Main(cols_numbered.leaf_label),
                            F::from_canonical_usize(CHUNK),
                        )],
                        F::from_canonical_usize(i),
                    ),
                    VirtualPairCol::single_main(cols_numbered.values[i]),
                ],
                count: VirtualPairCol::single_main(cols_numbered.temp_multiplicity[i]),
                argument_index: MEMORY_INTERFACE_BUS,
            })
            .collect()
    }
}
