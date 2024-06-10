use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::{columns::FlatHashCols, FlashHashChip};

impl Chip<F> for FlatHashChip<F> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_indices = self.hash_col_indices();
        let mut interactions = vec![];
        for i in 0..num_hashes {
            let mut fields = col_indices.hash_state_indices[i];
            fields.extend(col_indices.hash_chunk_indices[i]);
            fields.extend(col_indices.hash_output_indices[i]);

            interactions.push(Interaction {
                fields,
                count: VirtualPairCol::const_one(),
                argument_index: self.hashchip_bus_index(),
            });
        }

        interactions
    }
}
