use std::borrow::Borrow;

use afs_primitives::utils::not;
use afs_stark_backend::{air_builders::sub::SubAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS};
use p3_matrix::Matrix;

use super::{
    columns::{KeccakVmCols, NUM_KECCAK_VM_COLS},
    KECCAK_RATE_BYTES,
};

#[derive(Clone, Copy, Debug)]
pub struct KeccakVmAir {
    /// The index for the bus to send 8-bit XOR requests to.
    pub xor_bus_index: usize,
    // TODO: add configuration for enabling direct non-memory interactions
}

impl<F> BaseAir<F> for KeccakVmAir {
    fn width(&self) -> usize {
        NUM_KECCAK_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let [local, next] = [0, 1].map(|i| main.row_slice(i));
        let local: &KeccakVmCols<AB::Var> = (*local).borrow();
        let next: &KeccakVmCols<AB::Var> = (*next).borrow();

        // Not strictly necessary:
        builder
            .when_first_row()
            .assert_one(local.sponge.is_new_start);

        builder.assert_bool(local.io.is_opcode);
        // All rounds of a single permutation must have same is_opcode, clk, dst, e (src, a, c are only read on the 0-th round right now)
        let mut transition_builder = builder.when_transition();
        let mut round_builder =
            transition_builder.when(not::<AB>(local.inner.step_flags[NUM_ROUNDS - 1]));
        round_builder.assert_eq(local.io.is_opcode, next.io.is_opcode);
        round_builder.assert_eq(local.io.clk, next.io.clk);
        round_builder.assert_eq(local.io.e, next.io.e);
        round_builder.assert_eq(local.aux.dst, next.aux.dst);

        self.eval_keccak_f(builder);

        self.eval_interactions(builder, local);
    }
}

impl KeccakVmAir {
    pub fn new(xor_bus_index: usize) -> Self {
        Self { xor_bus_index }
    }

    /// Evaluate the keccak-f permutation constraints.
    ///
    /// WARNING: The keccak-f AIR columns **must** be the first columns in the main AIR.
    #[inline]
    pub fn eval_keccak_f<AB: AirBuilder>(&self, builder: &mut AB) {
        let keccak_f_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_PERM_COLS);
        keccak_f_air.eval(&mut sub_builder);
    }

    /// Keccak follows the 10*1 padding rule.
    /// See Section 5.1 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    /// Note this is the ONLY difference between Keccak and SHA-3
    pub fn constrain_padding<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let is_padding_byte = local.sponge.is_padding_byte;
        let block_bytes = &local.sponge.block_bytes;
        let remaining_len = local.remaining_len();
        let step_flags = &local.inner.step_flags;

        // is_padding_byte should all be boolean
        for &is_padding_byte in is_padding_byte.iter() {
            builder.assert_bool(is_padding_byte);
        }
        let mut num_padding_bytes = AB::Expr::zero();
        // is_padding_byte should transition from 0 to 1 only once and then stay 1
        for i in 1..KECCAK_RATE_BYTES {
            builder
                .when(is_padding_byte[i - 1])
                .assert_one(is_padding_byte[i]);
            num_padding_bytes = num_padding_bytes + is_padding_byte[i];
        }
        // is_padding_byte must stay the same on all rounds in a block
        let is_last_round = step_flags.last().unwrap();
        let is_not_last_round = not(is_last_round);
        for i in 0..KECCAK_RATE_BYTES {
            builder.when(is_not_last_round).assert_eq(
                local.sponge.is_padding_byte[i],
                next.sponge.is_padding_byte[i],
            );
        }

        // If final rate block of input, then last byte must be padding
        let is_final_block = is_padding_byte[KECCAK_RATE_BYTES - 1];

        // is_padding_byte must be consistent with remaining_len
        builder.when(is_final_block).assert_eq(
            remaining_len,
            AB::F::from_canonical_usize(KECCAK_RATE_BYTES) - num_padding_bytes,
        );
        // If this block is not final, when transitioning to next block, remaining len
        // must decrease by `KECCAK_RATE_BYTES`.
        builder
            .when(is_last_round)
            .when(not(is_final_block))
            .assert_eq(
                remaining_len - AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
                next.remaining_len(),
            );
        // To enforce that is_padding_byte must be set appropriately for an input, we require
        // the block before a new start to have padding
        builder
            .when(is_last_round)
            .when(next.is_new_start())
            .assert_one(is_final_block);
        // The chain above enforces that for an input, the remaining length must decrease by RATE
        // block-by-block until it reaches a final block with padding.

        // ====== Constrain the block_bytes are padded according to is_padding_byte =====

        // If the first padding byte is at the end of the block, then the block has a
        // single padding byte
        let has_single_padding_byte: AB::Expr =
            is_padding_byte[KECCAK_RATE_BYTES - 1] - is_padding_byte[KECCAK_RATE_BYTES - 2];

        // If the row has a single padding byte, then it must be the last byte with
        // value 0b10000001
        builder.when(has_single_padding_byte.clone()).assert_eq(
            block_bytes[KECCAK_RATE_BYTES - 1],
            AB::F::from_canonical_u8(0b10000001),
        );

        for i in 0..KECCAK_RATE_BYTES - 1 {
            let is_first_padding_byte: AB::Expr = {
                if i > 0 {
                    is_padding_byte[i] - is_padding_byte[i - 1]
                } else {
                    is_padding_byte[i].into()
                }
            };
            // If the row has multiple padding bytes, the first padding byte must be 0x80
            builder
                .when(not(has_single_padding_byte.clone()))
                .when(is_first_padding_byte.clone())
                .assert_eq(block_bytes[i], AB::F::from_canonical_u8(0x80));
            // If the row has multiple padding bytes, the other padding bytes
            // except the last one must be 0
            builder
                .when(is_padding_byte[i])
                .when(not(is_first_padding_byte)) // hence never when single padding byte
                .assert_zero(block_bytes[i]);
        }

        // If the row has multiple padding bytes, then the last byte must be 0x01
        builder
            .when(is_final_block)
            .when(not(has_single_padding_byte))
            .assert_eq(
                block_bytes[KECCAK_RATE_BYTES - 1],
                AB::F::from_canonical_u8(0x01),
            );
    }
}
