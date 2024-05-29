use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;

use super::{columns::KECCAK_COL_MAP, KeccakPermuteChip, NUM_U64_HASH_ELEMS, U64_LIMBS};

impl<F: PrimeField32> Chip<F> for KeccakPermuteChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: (0..NUM_U64_HASH_ELEMS)
                .flat_map(|i| {
                    (0..U64_LIMBS)
                        .map(|limb| {
                            let y = i / 5;
                            let x = i % 5;
                            KECCAK_COL_MAP.a_prime_prime_prime(y, x, limb)
                        })
                        .collect::<Vec<_>>()
                })
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(KECCAK_COL_MAP.is_real_digest),
            argument_index: self.bus_output_digest,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: KECCAK_COL_MAP
                .preimage
                .into_iter()
                .flatten()
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(KECCAK_COL_MAP.is_real_input),
            argument_index: self.bus_input,
        }]
    }
}
