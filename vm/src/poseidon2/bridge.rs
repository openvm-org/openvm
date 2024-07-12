use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::{BaseAir, PairCol, VirtualPairCol};
use p3_field::Field;
use poseidon2_air::poseidon2::columns::Poseidon2Cols;

use super::columns::Poseidon2ChipCols;
use super::Poseidon2Chip;
use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS};

/// Receives instructions from the CPU on the designated `POSEIDON2_BUS`, and sends both read and write requests to the memory chip.
/// Receives (clk, a, b, c, d, e, cmp)
impl<const WIDTH: usize, T: Field> AirBridge<T> for Poseidon2Chip<WIDTH, T> {
    fn receives(&self) -> Vec<Interaction<T>> {
        let indices: Vec<usize> = (0..self.width()).collect();
        let index_map = Poseidon2Cols::index_map(&self.air);
        let col_indices = Poseidon2ChipCols::from_slice(&indices, &index_map);
        let fields = col_indices
            .io
            .flatten()
            .into_iter()
            .skip(1)
            .map(VirtualPairCol::single_main)
            .collect();

        vec![Interaction {
            fields,
            count: VirtualPairCol::single_main(col_indices.io.is_alloc),
            argument_index: POSEIDON2_BUS,
        }]
    }

    fn sends(&self) -> Vec<Interaction<T>> {
        let chunks: usize = WIDTH / 2;
        let indices: Vec<usize> = (0..self.width()).collect();
        let index_map = Poseidon2Cols::index_map(&self.air);
        let col_indices = Poseidon2ChipCols::from_slice(&indices, &index_map);
        let mut interactions = vec![];
        // READ
        for i in 0..WIDTH {
            let memory_cycle = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.clk), T::one())],
                T::from_canonical_usize(i),
            );
            let address = VirtualPairCol::new(
                vec![(
                    PairCol::Main(if i < chunks {
                        col_indices.io.a
                    } else {
                        col_indices.io.b
                    }),
                    T::from_canonical_usize(1),
                )],
                T::from_canonical_usize(if i < chunks { i } else { i - chunks }),
            );

            let fields = vec![
                memory_cycle,
                VirtualPairCol::constant(T::from_bool(false)),
                VirtualPairCol::single_main(col_indices.io.d),
                address,
                VirtualPairCol::single_main(col_indices.internal.io.input[i]),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::single_main(col_indices.io.is_alloc),
                argument_index: MEMORY_BUS,
            });
        }

        // WRITE
        for i in 0..WIDTH {
            let memory_cycle = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.clk), T::one())],
                T::from_canonical_usize(i + WIDTH),
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
                VirtualPairCol::single_main(col_indices.internal.io.output[i]),
            ];

            let count = if i < chunks {
                VirtualPairCol::single_main(col_indices.io.is_alloc)
            } else {
                VirtualPairCol::diff_main(col_indices.io.is_alloc, col_indices.io.cmp)
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
