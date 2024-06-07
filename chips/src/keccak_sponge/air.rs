use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::columns::{
    KeccakSpongeCols, KECCAK_DIGEST_U16S, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    NUM_KECCAK_SPONGE_COLS,
};
use super::KeccakSpongeAir;

impl<F> BaseAir<F> for KeccakSpongeAir {
    fn width(&self) -> usize {
        NUM_KECCAK_SPONGE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakSpongeAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);
        let local: &KeccakSpongeCols<AB::Var> = (*local).borrow();
        let next: &KeccakSpongeCols<AB::Var> = (*next).borrow();

        // Each flag (full-input block, final block, padding byte or implied dummy flag)
        // must be boolean.
        let is_full_input_block = local.is_full_input_block;
        builder.assert_bool(is_full_input_block);

        let is_final_block = local.is_padding_byte[KECCAK_RATE_BYTES - 1];
        for &is_padding_byte in local.is_padding_byte.iter() {
            builder.assert_bool(is_padding_byte);
        }
        for i in 1..KECCAK_RATE_BYTES {
            builder
                .when(local.is_padding_byte[i - 1])
                .assert_one(local.is_padding_byte[i]);
        }

        // Ensure that full-input block and final block flags are not set to 1 at the
        // same time.
        builder.assert_zero(is_final_block * is_full_input_block);

        // If this is the first row, the original sponge state should be 0 and
        // already_absorbed_bytes = 0.
        let already_absorbed_bytes = local.already_absorbed_bytes;
        builder.when_first_row().assert_zero(already_absorbed_bytes);
        for &original_rate_elem in local.original_rate_u16s.iter() {
            builder.when_first_row().assert_zero(original_rate_elem);
        }
        for &original_capacity_elem in local.original_capacity_u16s.iter() {
            builder.when_first_row().assert_zero(original_capacity_elem);
        }

        // If this is a final block, the next row's original sponge state should be 0
        // and already_absorbed_bytes = 0.
        builder
            .when(is_final_block)
            .assert_zero(next.already_absorbed_bytes);
        for &original_rate_elem in next.original_rate_u16s.iter() {
            builder.when(is_final_block).assert_zero(original_rate_elem);
        }
        for &original_capacity_elem in next.original_capacity_u16s.iter() {
            builder
                .when(is_final_block)
                .assert_zero(original_capacity_elem);
        }

        // If this is a full-input block, the next row's "before" should match our
        // "after" state.
        for (current_bytes_after, &next_before) in local
            .updated_digest_state_bytes
            .chunks_exact(2)
            .zip(&next.original_rate_u16s[..KECCAK_DIGEST_U16S])
        {
            let current_after = (0..2).fold(AB::Expr::zero(), |acc, i| {
                acc + current_bytes_after[i] * AB::Expr::from_canonical_usize(1 << (8 * i))
            });
            builder
                .when(is_full_input_block)
                .assert_zero(next_before - current_after);
        }
        for (&current_after, &next_before) in local
            .partial_updated_state_u16s
            .iter()
            .zip(next.original_rate_u16s[KECCAK_DIGEST_U16S..].iter())
        {
            builder
                .when(is_full_input_block)
                .assert_zero(next_before - current_after);
        }
        for (&current_after, &next_before) in local
            .partial_updated_state_u16s
            .iter()
            .skip(KECCAK_RATE_U16S - KECCAK_DIGEST_U16S)
            .zip(next.original_capacity_u16s.iter())
        {
            builder
                .when(is_full_input_block)
                .assert_zero(next_before - current_after);
        }

        // If this is a full-input block, the next row's already_absorbed_bytes should
        // be ours plus `KECCAK_RATE_BYTES`.
        builder.when(is_full_input_block).assert_zero(
            already_absorbed_bytes + AB::Expr::from_canonical_usize(KECCAK_RATE_BYTES)
                - next.already_absorbed_bytes,
        );

        // If the first padding byte is at the end of the block, then the block has a
        // single padding byte
        let has_single_padding_byte = local.is_padding_byte[KECCAK_RATE_BYTES - 1]
            - local.is_padding_byte[KECCAK_RATE_BYTES - 2];

        // If the row has a single padding byte, then it must be the last byte with
        // value 0b10000001
        builder.when(has_single_padding_byte.clone()).assert_eq(
            local.block_bytes[KECCAK_RATE_BYTES - 1],
            AB::Expr::from_canonical_u8(0b10000001),
        );

        for i in 0..KECCAK_RATE_BYTES - 1 {
            let is_first_padding_byte = {
                if i > 0 {
                    local.is_padding_byte[i] - local.is_padding_byte[i - 1]
                } else {
                    local.is_padding_byte[i].into()
                }
            };
            // If the row has multiple padding bytes, the first padding byte must be 1
            builder
                .when(is_first_padding_byte.clone())
                .assert_one(local.block_bytes[i]);
            // If the row has multiple padding bytes, the other padding bytes
            // except the last one must be 0
            builder
                .when(local.is_padding_byte[i])
                .when_ne(is_first_padding_byte, AB::Expr::one())
                .assert_zero(local.block_bytes[i]);
        }

        // If the row has multiple padding bytes, then the last byte must be 0b10000000
        builder
            .when(is_final_block)
            .when_ne(has_single_padding_byte, AB::Expr::one())
            .assert_eq(
                local.block_bytes[KECCAK_RATE_BYTES - 1],
                AB::Expr::from_canonical_u8(0b10000000),
            );

        // TODO: Add back
        // // A dummy row is always followed by another dummy row, so the prover can't put
        // // dummy rows "in between" to avoid the above checks.
        // let is_dummy = AB::Expr::one() - is_full_input_block - is_final_block;
        // let next_is_final_block = next
        //     .is_final_input_len
        //     .iter()
        //     .fold(AB::Expr::zero(), |acc, &is_final_len| acc + is_final_len);
        // builder
        //     .when(is_dummy)
        //     .assert_zero(next.is_full_input_block + next_is_final_block);
    }
}
