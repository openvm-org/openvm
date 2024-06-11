use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::merkle_proof_col_map, MerkleProofAir};

// TODO: Replace keccak pseudo permutation with full hash
impl<F, const DEPTH: usize, const DIGEST_WIDTH: usize> AirBridge<F>
    for MerkleProofAir<DEPTH, DIGEST_WIDTH>
where
    F: PrimeField32,
{
    fn sends(&self) -> Vec<Interaction<F>> {
        let merkle_proof_col_map = merkle_proof_col_map::<DEPTH, DIGEST_WIDTH>();

        vec![Interaction {
            fields: merkle_proof_col_map
                .left_node
                .chunks_exact(2)
                .chain(merkle_proof_col_map.right_node.chunks(2))
                .map(|limbs| {
                    VirtualPairCol::new_main(
                        vec![
                            (limbs[0], F::one()),
                            (limbs[1], F::from_canonical_usize(1 << 8)),
                        ],
                        F::zero(),
                    )
                })
                .collect(),
            count: VirtualPairCol::single_main(merkle_proof_col_map.is_real),
            argument_index: self.bus_hash_input,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let merkle_proof_col_map = merkle_proof_col_map::<DEPTH, DIGEST_WIDTH>();

        vec![Interaction {
            fields: merkle_proof_col_map
                .output
                .chunks_exact(2)
                .map(|limbs| {
                    VirtualPairCol::new_main(
                        vec![
                            (limbs[0], F::one()),
                            (limbs[1], F::from_canonical_usize(1 << 8)),
                        ],
                        F::zero(),
                    )
                })
                .collect(),
            count: VirtualPairCol::single_main(merkle_proof_col_map.is_real),
            argument_index: self.bus_hash_output,
        }]
    }
}
