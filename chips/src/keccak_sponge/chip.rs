use afs_stark_backend::interaction::{Chip, Interaction};
use itertools::Itertools;
use p3_air::VirtualPairCol;
use p3_field::Field;

use super::{
    columns::{KECCAK_DIGEST_BYTES, KECCAK_RATE_BYTES, KECCAK_SPONGE_COL_MAP},
    KeccakSpongeChip,
};

// TODO: Add interaction with xor chip
impl<F: Field> Chip<F> for KeccakSpongeChip {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_map = KECCAK_SPONGE_COL_MAP;

        let is_real = VirtualPairCol::sum_main(vec![
            col_map.is_padding_byte[KECCAK_RATE_BYTES - 1],
            col_map.is_full_input_block,
        ]);
        [
            vec![Interaction {
                fields: (0..KECCAK_DIGEST_BYTES)
                    .map(|i| VirtualPairCol::single_main(col_map.updated_digest_state_bytes[i]))
                    .collect_vec(),
                count: is_real.clone(),
                argument_index: self.bus_output,
            }],
            // col_map
            //     .block_bytes
            //     .chunks(4)
            //     .zip(col_map.original_rate_u16s.chunks(2))
            //     .map(|(block, rate)| {
            //         let vc1 = {
            //             let column_weights = block
            //                 .iter()
            //                 .enumerate()
            //                 .map(|(i, &c)| (c, F::from_canonical_usize(1 << (8 * i))))
            //                 .collect_vec();
            //             VirtualPairCol::new_main(column_weights, F::zero())
            //         };
            //         let vc2 = {
            //             let column_weights = rate
            //                 .iter()
            //                 .enumerate()
            //                 .map(|(i, &c)| (c, F::from_canonical_usize(1 << (16 * i))))
            //                 .collect_vec();
            //             VirtualPairCol::new_main(column_weights, F::zero())
            //         };
            //         Interaction {
            //             fields: vec![vc1, vc2],
            //             count: is_real.clone(),
            //             argument_index: self.bus_xor_input,
            //         }
            //     })
            //     .collect_vec(),
            vec![Interaction {
                fields: col_map
                    .xored_rate_u16s
                    .into_iter()
                    .chain(col_map.original_capacity_u16s)
                    .map(VirtualPairCol::single_main)
                    .collect(),
                count: is_real.clone(),
                argument_index: self.bus_permute_input,
            }],
        ]
        .concat()
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let col_map = KECCAK_SPONGE_COL_MAP;

        let is_real = VirtualPairCol::sum_main(vec![
            col_map.is_padding_byte[KECCAK_RATE_BYTES - 1],
            col_map.is_full_input_block,
        ]);
        [
            // TODO: Only send non padding bytes. Interaction field should be
            //       is_padding_byte[i] * block_bytes[i] but requires degree 2 fields
            vec![Interaction {
                fields: (0..KECCAK_RATE_BYTES)
                    .map(|i| VirtualPairCol::single_main(col_map.block_bytes[i]))
                    .collect_vec(),
                count: is_real.clone(),
                argument_index: self.bus_input,
            }],
            // col_map
            //     .xored_rate_u16s
            //     .chunks(2)
            //     .map(|rate| {
            //         let column_weights = rate
            //             .iter()
            //             .enumerate()
            //             .map(|(i, &c)| (c, F::from_canonical_usize(1 << (16 * i))))
            //             .collect_vec();
            //         Interaction {
            //             fields: vec![VirtualPairCol::new_main(column_weights, F::zero())],
            //             count: is_real.clone(),
            //             argument_index: self.bus_xor_output,
            //         }
            //     })
            //     .collect_vec(),
            vec![Interaction {
                // We recover the 16-bit digest limbs from their corresponding bytes,
                // and then append them to the rest of the updated state limbs.
                fields: col_map
                    .updated_digest_state_bytes
                    .chunks(2)
                    .map(|cols| {
                        let column_weights = cols
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| (c, F::from_canonical_usize(1 << (8 * i))))
                            .collect_vec();
                        VirtualPairCol::new_main(column_weights, F::zero())
                    })
                    .chain(
                        col_map
                            .partial_updated_state_u16s
                            .into_iter()
                            .map(VirtualPairCol::single_main),
                    )
                    .collect_vec(),
                count: is_real.clone(),
                argument_index: self.bus_permute_output,
            }],
        ]
        .concat()
    }
}
