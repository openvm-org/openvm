use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::{columns::FlatHashCols, FlatHashAir};

impl<F: Field> AirBridge<F> for FlatHashAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_indices =
            FlatHashCols::<F>::hash_col_indices(self.page_width, self.hash_width, self.hash_rate);
        let num_hashes = self.page_width / self.hash_rate;
        let mut interactions = vec![];
        for i in 0..num_hashes {
            let fields: Vec<_> = col_indices.hash_state_indices[i]
                .iter()
                .chain(&col_indices.hash_chunk_indices[i])
                .chain(&col_indices.hash_output_indices[i])
                .cloned()
                .map(VirtualPairCol::single_main)
                .collect();

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::single_main(col_indices.is_alloc_index),
                argument_index: self.hash_chip_bus_index,
            });
        }

        interactions
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let fields = (1..self.page_width + 1)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        vec![Interaction {
            fields,
            count: VirtualPairCol::single_main(self.hash_chip_bus_index),
            argument_index: self.bus_index,
        }]
    }
}
