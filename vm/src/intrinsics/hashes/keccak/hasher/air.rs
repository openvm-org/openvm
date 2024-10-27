use std::borrow::Borrow;

use ax_circuit_primitives::{utils::not, xor::XorBus};
use ax_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS};
use p3_matrix::Matrix;

use super::{
    columns::{KeccakVmCols, NUM_KECCAK_VM_COLS},
    KECCAK_RATE_BYTES,
};
use crate::{arch::ExecutionBridge, system::memory::offline_checker::MemoryBridge};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send 8-bit XOR requests to.
    pub xor_bus: XorBus,
    // TODO: add configuration for enabling direct non-memory interactions
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakVmAir {}
impl<F> PartitionedBaseAir<F> for KeccakVmAir {}
impl<F> BaseAir<F> for KeccakVmAir {
    fn width(&self) -> usize {
        NUM_KECCAK_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &KeccakVmCols<AB::Var> = (*local).borrow();
        let next: &KeccakVmCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.sponge.is_new_start);
        builder.assert_eq(
            local.sponge.is_new_start,
            local.sponge.is_new_start * local.is_first_round(),
        );
        builder.assert_eq(
            local.opcode.is_enabled_first_round,
            local.opcode.is_enabled * local.is_first_round(),
        );
        // Not strictly necessary:
        builder
            .when_first_row()
            .assert_one(local.sponge.is_new_start);

        self.eval_keccak_f(builder);
        self.constrain_padding(builder, local, next);
        self.constrain_consistency_across_rounds(builder, local, next);

        let mem = &local.mem_oc;
        // Interactions:
        self.constrain_absorb(builder, local, next);
        let start_read_timestamp = self.eval_opcode_interactions(builder, local, &mem.op_reads);
        let start_write_timestamp =
            self.constrain_input_read(builder, local, start_read_timestamp, &mem.absorb_reads);
        self.constrain_output_write(
            builder,
            local,
            start_write_timestamp.clone(),
            &mem.digest_writes,
        );

        self.constrain_block_transition(builder, local, next, start_write_timestamp);
    }
}

impl KeccakVmAir {
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

    /// Many columns are expected to be the same between rounds and only change per-block.
    pub fn constrain_consistency_across_rounds<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let mut transition_builder = builder.when_transition();
        let mut round_builder = transition_builder.when(not(local.is_last_round()));
        // Opcode columns
        local.opcode.assert_eq(&mut round_builder, next.opcode);
    }

    pub fn constrain_block_transition<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
    ) {
        // When we transition between blocks, if the next block isn't a new block
        // (this means it's not receiving a new opcode or starting a dummy block)
        // then we want _parts_ of opcode instruction to stay the same
        // between blocks.
        let mut block_transition = builder.when(local.is_last_round() * not(next.is_new_start()));
        block_transition.assert_eq(local.opcode.is_enabled, next.opcode.is_enabled);
        // dst is only going to be used for writes in the last input block
        block_transition.assert_eq(local.opcode.dst, next.opcode.dst);
        // needed for memory reads
        block_transition.assert_eq(local.opcode.e, next.opcode.e);
        // these are not used and hence not necessary, but putting for safety until performance becomes an issue:
        block_transition.assert_eq(local.opcode.a, next.opcode.a);
        block_transition.assert_eq(local.opcode.b, next.opcode.b);
        block_transition.assert_eq(local.opcode.c, next.opcode.c);
        block_transition.assert_eq(local.opcode.d, next.opcode.d);

        // Move the src pointer over based on the number of bytes read.
        // This should always be RATE_BYTES since it's a non-final block.
        // TODO: depends on WORD_SIZE
        block_transition.assert_eq(
            next.opcode.src,
            local.opcode.src + AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
        );
        // Advance timestamp by the number of memory accesses from reading
        // `dst, src, len` and block input bytes.
        block_transition.assert_eq(next.opcode.start_timestamp, start_write_timestamp);
        block_transition.assert_eq(
            next.opcode.len,
            local.opcode.len - AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
        );
        // Padding transition is constrained in `constrain_padding`.
    }

    /// Keccak follows the 10*1 padding rule.
    /// See Section 5.1 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    /// Note this is the ONLY difference between Keccak and SHA-3
    ///
    /// Constrains padding constraints and length between rounds and
    /// between blocks. Padding logic is tied to constraints on `is_new_start`.
    pub fn constrain_padding<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let is_padding_byte = local.sponge.is_padding_byte;
        let block_bytes = &local.sponge.block_bytes;
        let remaining_len = local.remaining_len();

        // is_padding_byte should all be boolean
        for &is_padding_byte in is_padding_byte.iter() {
            builder.assert_bool(is_padding_byte);
        }
        // is_padding_byte should transition from 0 to 1 only once and then stay 1
        for i in 1..KECCAK_RATE_BYTES {
            builder
                .when(is_padding_byte[i - 1])
                .assert_one(is_padding_byte[i]);
        }
        // is_padding_byte must stay the same on all rounds in a block
        // we use next instead of local.step_flags.last() because the last row of the trace overall may not
        // end on a last round
        let is_last_round = next.inner.step_flags[0];
        let is_not_last_round = not(is_last_round);
        for i in 0..KECCAK_RATE_BYTES {
            builder.when(is_not_last_round.clone()).assert_eq(
                local.sponge.is_padding_byte[i],
                next.sponge.is_padding_byte[i],
            );
        }

        let num_padding_bytes = local
            .sponge
            .is_padding_byte
            .iter()
            .fold(AB::Expr::zero(), |a, &b| a + b);

        // If final rate block of input, then last byte must be padding
        let is_final_block = is_padding_byte[KECCAK_RATE_BYTES - 1];

        // is_padding_byte must be consistent with remaining_len
        builder.when(is_final_block).assert_eq(
            remaining_len,
            AB::Expr::from_canonical_usize(KECCAK_RATE_BYTES) - num_padding_bytes,
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
        // Make sure there are not repeated padding blocks
        builder
            .when(is_last_round)
            .when(is_final_block)
            .assert_one(next.is_new_start());
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

        let has_multiple_padding_bytes: AB::Expr = not(has_single_padding_byte.clone());
        for i in 0..KECCAK_RATE_BYTES - 1 {
            let is_first_padding_byte: AB::Expr = {
                if i > 0 {
                    is_padding_byte[i] - is_padding_byte[i - 1]
                } else {
                    is_padding_byte[i].into()
                }
            };
            // If the row has multiple padding bytes, the first padding byte must be 0x01
            // because the padding 1*0 is *little-endian*
            builder
                .when(has_multiple_padding_bytes.clone())
                .when(is_first_padding_byte.clone())
                .assert_eq(block_bytes[i], AB::F::from_canonical_u8(0x01));
            // If the row has multiple padding bytes, the other padding bytes
            // except the last one must be 0
            builder
                .when(is_padding_byte[i])
                .when(not::<AB::Expr>(is_first_padding_byte)) // hence never when single padding byte
                .assert_zero(block_bytes[i]);
        }

        // If the row has multiple padding bytes, then the last byte must be 0x80
        // because the padding *01 is *little-endian*
        builder
            .when(is_final_block)
            .when(has_multiple_padding_bytes)
            .assert_eq(
                block_bytes[KECCAK_RATE_BYTES - 1],
                AB::F::from_canonical_u8(0x80),
            );
    }
}
