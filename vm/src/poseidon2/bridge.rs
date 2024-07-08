use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::{BaseAir, PairCol, VirtualPairCol};
use p3_field::Field;
use poseidon2::poseidon2::columns::Poseidon2Cols;

use super::columns::Poseidon2ChipCols;
use super::Poseidon2Chip;
use crate::cpu::MEMORY_BUS;

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl<const WIDTH: usize, T: Field> AirBridge<T> for Poseidon2Chip<WIDTH, T> {
    fn receives(&self) -> Vec<Interaction<T>> {
        vec![]
    }

    fn sends(&self) -> Vec<Interaction<T>> {
        let indices: Vec<usize> = (0..self.width()).collect();
        let index_map = Poseidon2Cols::index_map(&self.air);
        let col_indices = Poseidon2ChipCols::from_slice(&indices, &index_map);
        let mut interactions = vec![];
        // READ
        for i in 0..16 {
            let memory_cycle = VirtualPairCol::new(
                vec![(
                    PairCol::Main(col_indices.io.clk),
                    T::from_canonical_usize(20),
                )],
                T::from_canonical_usize(i),
            );
            let address = VirtualPairCol::new(
                vec![(
                    PairCol::Main(if i < 8 {
                        col_indices.io.a
                    } else {
                        col_indices.io.b
                    }),
                    T::from_canonical_usize(1),
                )],
                T::from_canonical_usize(i),
            );

            let fields = vec![
                memory_cycle,
                VirtualPairCol::constant(T::from_bool(false)),
                VirtualPairCol::single_main(col_indices.io.d),
                address,
                VirtualPairCol::single_main(col_indices.aux.io.input[i]),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::single_main(col_indices.io.is_alloc),
                argument_index: MEMORY_BUS,
            });
        }

        // WRITE
        for i in 0..16 {
            let memory_cycle = VirtualPairCol::new(
                vec![(
                    PairCol::Main(col_indices.io.clk),
                    T::from_canonical_usize(20),
                )],
                T::from_canonical_usize(i + 16),
            );
            let address = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.c), T::from_canonical_usize(1))],
                T::from_canonical_usize(i),
            );

            let fields = vec![
                memory_cycle,
                VirtualPairCol::constant(T::from_bool(true)),
                VirtualPairCol::single_main(col_indices.io.e),
                address,
                VirtualPairCol::single_main(col_indices.aux.io.output[i]),
            ];

            let count = if i < 8 {
                VirtualPairCol::single_main(col_indices.io.is_alloc)
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
