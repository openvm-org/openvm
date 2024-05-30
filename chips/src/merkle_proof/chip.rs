use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::MERKLE_PROOF_COL_MAP, MerkleProofChip};

impl<F: PrimeField32> Chip<F> for MerkleProofChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: MERKLE_PROOF_COL_MAP
                .left_node
                .into_iter()
                .chain(MERKLE_PROOF_COL_MAP.right_node)
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(MERKLE_PROOF_COL_MAP.is_real),
            argument_index: self.bus_hash_input,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        // vec![Interaction {
        //     fields: MERKLE_PROOF_COL_MAP
        //         .output
        //         .into_iter()
        //         .flatten()
        //         .map(VirtualPairCol::single_main)
        //         .collect(),
        //     count: VirtualPairCol::single_main(MERKLE_PROOF_COL_MAP.is_real),
        //     argument_index: self.bus_hash_output,
        // }]
        vec![]
    }
}
