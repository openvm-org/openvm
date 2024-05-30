use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;
use p3_keccak_air::U64_LIMBS;

use super::{columns::KECCAK_PERMUTE_COL_MAP, KeccakPermuteChip, NUM_U64_HASH_ELEMS};

impl<F: PrimeField32> Chip<F> for KeccakPermuteChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: (0..NUM_U64_HASH_ELEMS)
                .flat_map(|i| {
                    (0..U64_LIMBS)
                        .map(|limb| {
                            // TODO: Wrong, should be the other way around, check latest p3
                            let y = i % 5;
                            let x = i / 5;
                            KECCAK_PERMUTE_COL_MAP
                                .keccak
                                .a_prime_prime_prime(y, x, limb)
                        })
                        .collect::<Vec<_>>()
                })
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(KECCAK_PERMUTE_COL_MAP.is_real_output),
            argument_index: self.bus_output,
        }]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: KECCAK_PERMUTE_COL_MAP
                .keccak
                .preimage
                .into_iter()
                .flatten()
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(KECCAK_PERMUTE_COL_MAP.is_real_input),
            argument_index: self.bus_input,
        }]
    }
}
