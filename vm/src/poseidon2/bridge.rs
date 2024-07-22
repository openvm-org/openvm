use p3_air::{BaseAir, PairCol, VirtualPairCol};
use p3_field::Field;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use poseidon2_air::poseidon2::columns::Poseidon2Cols;

use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS};

use super::columns::Poseidon2ChipCols;
use super::Poseidon2Chip;

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

        let mut timestamp_offset = 0;
        // read addresses
        for i in 0..3 {
            let timestamp = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.clk), T::one())],
                T::from_canonical_usize(timestamp_offset),
            );
            timestamp_offset += 1;

            let address_col = [col_indices.io.a, col_indices.io.b, col_indices.io.c][i];

            let fields = vec![
                timestamp,
                VirtualPairCol::constant(T::from_bool(false)),
                VirtualPairCol::single_main(col_indices.io.d),
                VirtualPairCol::single_main(address_col),
                VirtualPairCol::single_main(col_indices.aux.addresses[i]),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::diff_main(
                    col_indices.io.is_alloc,
                    col_indices.aux.d_is_zero,
                ),
                argument_index: MEMORY_BUS,
            });
        }
        // READ
        for i in 0..WIDTH {
            let timestamp = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.clk), T::one())],
                T::from_canonical_usize(timestamp_offset),
            );
            timestamp_offset += 1;
            let address = VirtualPairCol::new(
                vec![(
                    PairCol::Main(if i < chunks {
                        col_indices.aux.addresses[0]
                    } else {
                        col_indices.aux.addresses[1]
                    }),
                    T::from_canonical_usize(1),
                )],
                T::from_canonical_usize(if i < chunks { i } else { i - chunks }),
            );

            let fields = vec![
                timestamp,
                VirtualPairCol::constant(T::from_bool(false)),
                VirtualPairCol::single_main(col_indices.io.e),
                address,
                VirtualPairCol::single_main(col_indices.aux.internal.io.input[i]),
            ];

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::single_main(col_indices.io.is_alloc),
                argument_index: MEMORY_BUS,
            });
        }

        // WRITE
        for i in 0..WIDTH {
            let timestamp = VirtualPairCol::new(
                vec![(PairCol::Main(col_indices.io.clk), T::one())],
                T::from_canonical_usize(timestamp_offset),
            );
            timestamp_offset += 1;
            let address = VirtualPairCol::new(
                vec![(
                    PairCol::Main(col_indices.aux.addresses[2]),
                    T::from_canonical_usize(1),
                )],
                T::from_canonical_usize(i),
            );

            let fields = vec![
                timestamp,
                VirtualPairCol::constant(T::from_bool(true)),
                VirtualPairCol::single_main(col_indices.io.e),
                address,
                VirtualPairCol::single_main(col_indices.aux.internal.io.output[i]),
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
