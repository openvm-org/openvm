use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::merkle_proof_col_map, MerkleProofChip};

impl<F: PrimeField32, const DEPTH: usize> Chip<F> for MerkleProofChip<DEPTH> {
    fn sends(&self) -> Vec<Interaction<F>> {
        let merkle_proof_col_map = merkle_proof_col_map::<DEPTH>();

        vec![Interaction {
            fields: merkle_proof_col_map
                .left_node
                .into_iter()
                .chain(merkle_proof_col_map.right_node)
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(merkle_proof_col_map.is_real),
            argument_index: self.bus_hash_input,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let merkle_proof_col_map = merkle_proof_col_map::<DEPTH>();

        vec![Interaction {
            fields: merkle_proof_col_map
                .output
                .into_iter()
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(merkle_proof_col_map.is_real),
            argument_index: self.bus_hash_output,
        }]
    }
}
