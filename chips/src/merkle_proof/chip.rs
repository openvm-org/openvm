use std::iter::once;

use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::merkle_proof_col_map, MerkleProofChip};
use crate::keccak_sponge::columns::KECCAK_RATE_BYTES;

impl<F, const DEPTH: usize, const DIGEST_WIDTH: usize> Chip<F>
    for MerkleProofChip<DEPTH, DIGEST_WIDTH>
where
    F: PrimeField32,
{
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_map = merkle_proof_col_map::<DEPTH, DIGEST_WIDTH>();

        vec![Interaction {
            fields: once(VirtualPairCol::constant(F::zero()))
                .chain(
                    col_map
                        .left_node
                        .into_iter()
                        .chain(col_map.right_node.into_iter())
                        .map(|elem| VirtualPairCol::single_main(elem))
                        // TODO: Don't send padding bytes
                        .chain((2 * DIGEST_WIDTH..KECCAK_RATE_BYTES).map(|i| {
                            VirtualPairCol::constant({
                                if i == 2 * DIGEST_WIDTH {
                                    F::one()
                                } else if i == KECCAK_RATE_BYTES - 1 {
                                    F::from_canonical_u8(0b10000000)
                                } else {
                                    F::zero()
                                }
                            })
                        })),
                )
                .collect(),
            count: VirtualPairCol::single_main(col_map.is_real),
            argument_index: self.bus_hash_input,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let col_map = merkle_proof_col_map::<DEPTH, DIGEST_WIDTH>();

        vec![Interaction {
            fields: col_map
                .output
                .into_iter()
                .map(|elem| VirtualPairCol::single_main(elem))
                .collect(),
            count: VirtualPairCol::single_main(col_map.is_real),
            argument_index: self.bus_hash_output,
        }]
    }
}
