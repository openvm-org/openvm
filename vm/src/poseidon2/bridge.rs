use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::columns::Poseidon2ChipCols;
use super::Poseidon2Chip;

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl<const WIDTH: usize, T: Field> AirBridge<T> for Poseidon2Chip<WIDTH, T> {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![]
    }

    fn sends(&self) -> Vec<Interaction<T>> {
        let col_indices = Poseidon2ChipCols::from_slice(0..self.get_width(), self);
        let mut interactions = vec![];
        // READ
        for i in 0..16 {
            let memory_cycle = VirtualPairCol::new(
                vec![(
                    PairCol::Main(col_indices.io.clk),
                    F::from_canonical_usize(20),
                )],
                F::from_canonical_usize(i),
            );
            let address = VirtualPairCol::new(
                vec![(PairCol::Main(i < 8 ? col_indices.io.a : col_indices.io.b),)],
                F::from_canonical_usize(i),
            );

            let mut fields = vec![
                memory_cycle,
                VirtualPairCol::constant(F::from_bool(false)),
                VirtualPairCol::single_main(col_indices.io.d),
                address,
                VirtualPairCol::single_main(col_indices.aux.io.input[i]),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::constant(F::one()),
                argument_index: MEMORY_BUS,
            });
        }

        // WRITE
        for i in 0..16 {
            let memory_cycle = VirtualPairCol::new(
                vec![(
                    PairCol::Main(col_indices.io.clk),
                    F::from_canonical_usize(20),
                )],
                F::from_canonical_usize(i + 16),
            );
            let address = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.c),)],
                F::from_canonical_usize(i),
            );

            let mut fields = vec![
                memory_cycle,
                VirtualPairCol::constant(F::from_bool(true)),
                VirtualPairCol::single_main(col_indices.io.e),
                address,
                VirtualPairCol::single_main(col_indices.aux.io.output[i]),
            ];

            let count = if (i < 8) {
                VirtualPairCol::constant(F::one())
            } else {
                VirtualPairCol::single_main(col_indices.io.cmp)
            };

            interactions.push(Interaction {
                fields,
                count,
                argument_index: MEMORY_BUS,
            });
        }

        interactions
    }
}
