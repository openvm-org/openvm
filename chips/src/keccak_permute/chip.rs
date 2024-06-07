use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField32;
use p3_keccak_air::U64_LIMBS;

use super::{columns::KECCAK_PERMUTE_COL_MAP, KeccakPermuteChip};

impl<F: PrimeField32> Chip<F> for KeccakPermuteChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_map = KECCAK_PERMUTE_COL_MAP;

        vec![Interaction {
            fields: (0..25)
                .flat_map(|i| {
                    (0..U64_LIMBS)
                        .map(|limb| {
                            // TODO: Wrong, should be the other way around, check latest p3
                            let y = i % 5;
                            let x = i / 5;
                            col_map.keccak.a_prime_prime_prime(y, x, limb)
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
        let col_map = KECCAK_PERMUTE_COL_MAP;

        vec![Interaction {
            fields: col_map
                .keccak
                .preimage
                .into_iter()
                .flatten()
                .flatten()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(col_map.is_real_input),
            argument_index: self.bus_input,
        }]
    }
}
