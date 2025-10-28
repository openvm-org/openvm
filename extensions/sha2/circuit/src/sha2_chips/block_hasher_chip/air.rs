use std::{cmp::max, iter::once, marker::PhantomData};

use ndarray::s;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    encoder::Encoder,
    utils::{not, select},
    SubAir,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{
    big_sig0_field, big_sig1_field, ch_field, compose, maj_field, small_sig0_field,
    small_sig1_field,
};
use crate::{
    constraint_word_addition, word_into_u16_limbs, Sha2BlockHasherConfig, Sha2DigestColsRef,
    Sha2RoundColsRef,
};

/// Expects the message to be padded to a multiple of C::BLOCK_WORDS * C::WORD_BITS bits
#[derive(Clone, Debug)]
pub struct Sha2BlockHasherAir<C: Sha2BlockHasherConfig> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub row_idx_encoder: Encoder,
    /// Internal bus for self-interactions in this AIR.
    bus: PermutationCheckBus,
    _phantom: PhantomData<C>,
}

impl<C: Sha2BlockHasherConfig> Sha2BlockHasherAir<C> {
    pub fn new(bitwise_lookup_bus: BitwiseOperationLookupBus, self_bus_idx: BusIndex) -> Self {
        Self {
            bitwise_lookup_bus,
            row_idx_encoder: Encoder::new(C::ROWS_PER_BLOCK + 1, 2, false), /* + 1 for dummy
                                                                             *   (padding) rows */
            bus: PermutationCheckBus::new(self_bus_idx),
            _phantom: PhantomData,
        }
    }
}

impl<F, C: Sha2BlockHasherConfig> BaseAirWithPublicValues<F> for Sha2BlockHasherAir<C> {}
impl<F, C: Sha2BlockHasherConfig> PartitionedBaseAir<F> for Sha2BlockHasherAir<C> {}
impl<F, C: Sha2BlockHasherConfig> BaseAir<F> for Sha2BlockHasherAir<C> {
    fn width(&self) -> usize {
        C::WIDTH
    }
}

impl<AB: InteractionBuilder, C: Sha2BlockHasherConfig> Air<AB> for Sha2BlockHasherAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.eval_row(builder);
        self.eval_transitions(builder);
    }
}

impl<C: Sha2BlockHasherConfig> Sha2BlockHasherAir<C> {
    /// Implements the single row constraints (i.e. imposes constraints only on local)
    /// Implements some sanity constraints on the row index, flags, and work variables
    fn eval_row<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        // Doesn't matter which column struct we use here as we are only interested in the common
        // columns
        let local_cols: Sha2DigestColsRef<AB::Var> =
            Sha2DigestColsRef::from::<C>(&local[..C::DIGEST_WIDTH]);
        let flags = &local_cols.flags;
        builder.assert_bool(*flags.is_round_row);
        builder.assert_bool(*flags.is_first_4_rows);
        builder.assert_bool(*flags.is_digest_row);
        builder.assert_bool(*flags.is_round_row + *flags.is_digest_row);
        builder.assert_bool(*flags.is_last_block);

        self.row_idx_encoder
            .eval(builder, local_cols.flags.row_idx.to_slice().unwrap());
        builder.assert_one(self.row_idx_encoder.contains_flag_range::<AB>(
            local_cols.flags.row_idx.to_slice().unwrap(),
            0..=C::ROWS_PER_BLOCK,
        ));
        builder.assert_eq(
            self.row_idx_encoder
                .contains_flag_range::<AB>(local_cols.flags.row_idx.to_slice().unwrap(), 0..=3),
            *flags.is_first_4_rows,
        );
        builder.assert_eq(
            self.row_idx_encoder.contains_flag_range::<AB>(
                local_cols.flags.row_idx.to_slice().unwrap(),
                0..=C::ROUND_ROWS - 1,
            ),
            *flags.is_round_row,
        );
        builder.assert_eq(
            self.row_idx_encoder.contains_flag::<AB>(
                local_cols.flags.row_idx.to_slice().unwrap(),
                &[C::ROUND_ROWS],
            ),
            *flags.is_digest_row,
        );
        // If padding row we want the row_idx to be C::ROWS_PER_BLOCK
        builder.assert_eq(
            self.row_idx_encoder.contains_flag::<AB>(
                local_cols.flags.row_idx.to_slice().unwrap(),
                &[C::ROWS_PER_BLOCK],
            ),
            flags.is_padding_row(),
        );

        // Constrain a, e, being composed of bits: we make sure a and e are always in the same place
        // in the trace matrix Note: this has to be true for every row, even padding rows
        for i in 0..C::ROUNDS_PER_ROW {
            for j in 0..C::WORD_BITS {
                builder.assert_bool(local_cols.hash.a[[i, j]]);
                builder.assert_bool(local_cols.hash.e[[i, j]]);
            }
        }
    }

    /// Implements constraints for a digest row that ensure proper state transitions between blocks
    /// This validates that:
    /// The work variables are correctly initialized for the next message block
    /// For the last message block, the initial state matches SHA_H constants
    fn eval_digest_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: Sha2RoundColsRef<AB::Var>,
        next: Sha2DigestColsRef<AB::Var>,
    ) {
        // Check that if this is the last row of a message or an inpadding row, the hash should be
        // the [SHA_H]
        for i in 0..C::ROUNDS_PER_ROW {
            let a = next.hash.a.row(i).mapv(|x| x.into()).to_vec();
            let e = next.hash.e.row(i).mapv(|x| x.into()).to_vec();

            for j in 0..C::WORD_U16S {
                let a_limb = compose::<AB::Expr>(&a[j * 16..(j + 1) * 16], 1);
                let e_limb = compose::<AB::Expr>(&e[j * 16..(j + 1) * 16], 1);

                // If it is a padding row or the last row of a message, the `hash` should be the
                // [SHA_H]
                builder
                    .when(
                        next.flags.is_padding_row()
                            + *next.flags.is_last_block * *next.flags.is_digest_row,
                    )
                    .assert_eq(
                        a_limb,
                        AB::Expr::from_canonical_u32(
                            word_into_u16_limbs::<C>(C::get_h()[C::ROUNDS_PER_ROW - i - 1])[j],
                        ),
                    );

                builder
                    .when(
                        next.flags.is_padding_row()
                            + *next.flags.is_last_block * *next.flags.is_digest_row,
                    )
                    .assert_eq(
                        e_limb,
                        AB::Expr::from_canonical_u32(
                            word_into_u16_limbs::<C>(C::get_h()[C::ROUNDS_PER_ROW - i + 3])[j],
                        ),
                    );
            }
        }

        // Check if last row of a non-last block, the `hash` should be equal to the final hash of
        // the current block
        for i in 0..C::ROUNDS_PER_ROW {
            let prev_a = next.hash.a.row(i).mapv(|x| x.into()).to_vec();
            let prev_e = next.hash.e.row(i).mapv(|x| x.into()).to_vec();
            let cur_a = next
                .final_hash
                .row(C::ROUNDS_PER_ROW - i - 1)
                .mapv(|x| x.into());

            let cur_e = next
                .final_hash
                .row(C::ROUNDS_PER_ROW - i + 3)
                .mapv(|x| x.into());
            for j in 0..C::WORD_U8S {
                let prev_a_limb = compose::<AB::Expr>(&prev_a[j * 8..(j + 1) * 8], 1);
                let prev_e_limb = compose::<AB::Expr>(&prev_e[j * 8..(j + 1) * 8], 1);

                builder
                    .when(not(*next.flags.is_last_block) * *next.flags.is_digest_row)
                    .assert_eq(prev_a_limb, cur_a[j].clone());

                builder
                    .when(not(*next.flags.is_last_block) * *next.flags.is_digest_row)
                    .assert_eq(prev_e_limb, cur_e[j].clone());
            }
        }

        // Assert that the previous hash + work vars == final hash.
        // That is, `next.prev_hash[i] + local.work_vars[i] == next.final_hash[i]`
        // where addition is done modulo 2^32
        for i in 0..C::HASH_WORDS {
            let mut carry = AB::Expr::ZERO;
            for j in 0..C::WORD_U16S {
                let work_var_limb = if i < C::ROUNDS_PER_ROW {
                    compose::<AB::Expr>(
                        local
                            .work_vars
                            .a
                            .slice(s![C::ROUNDS_PER_ROW - 1 - i, j * 16..(j + 1) * 16])
                            .as_slice()
                            .unwrap(),
                        1,
                    )
                } else {
                    compose::<AB::Expr>(
                        local
                            .work_vars
                            .e
                            .slice(s![C::ROUNDS_PER_ROW + 3 - i, j * 16..(j + 1) * 16])
                            .as_slice()
                            .unwrap(),
                        1,
                    )
                };
                let final_hash_limb = compose::<AB::Expr>(
                    next.final_hash
                        .slice(s![i, j * 2..(j + 1) * 2])
                        .as_slice()
                        .unwrap(),
                    8,
                );

                carry = AB::Expr::from(AB::F::from_canonical_u32(1 << 16).inverse())
                    * (next.prev_hash[[i, j]] + work_var_limb + carry - final_hash_limb);
                builder
                    .when(*next.flags.is_digest_row)
                    .assert_bool(carry.clone());
            }
            // constrain the final hash limbs two at a time since we can do two checks per
            // interaction
            for chunk in next.final_hash.row(i).as_slice().unwrap().chunks(2) {
                self.bitwise_lookup_bus
                    .send_range(chunk[0], chunk[1])
                    .eval(builder, *next.flags.is_digest_row);
            }
        }
    }

    fn eval_transitions<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // Doesn't matter what column structs we use here
        let local_cols: Sha2RoundColsRef<AB::Var> =
            Sha2RoundColsRef::from::<C>(&local[..C::ROUND_WIDTH]);
        let next_cols: Sha2RoundColsRef<AB::Var> =
            Sha2RoundColsRef::from::<C>(&next[..C::ROUND_WIDTH]);

        let local_is_padding_row = local_cols.flags.is_padding_row();
        // Note that there will always be a padding row in the trace since the unpadded height is a
        // multiple of 17 (SHA-256) or 21 (SHA-512, SHA-384). So the next row is padding iff the
        // current block is the last block in the trace.
        let next_is_padding_row = next_cols.flags.is_padding_row();

        // We check that the very last block has `is_last_block` set to true, which guarantees that
        // there is at least one complete message. If other digest rows have `is_last_block` set to
        // true, then the trace will be interpreted as containing multiple messages.
        builder
            .when(next_is_padding_row.clone())
            .when(*next_cols.flags.is_digest_row)
            .assert_one(*next_cols.flags.is_last_block);
        // If we are in a round row, the next row cannot be a padding row
        builder
            .when(*local_cols.flags.is_round_row)
            .assert_zero(next_is_padding_row.clone());
        // The first row must be a round row
        builder
            .when_first_row()
            .assert_one(*local_cols.flags.is_round_row);
        // If we are in a padding row, the next row must also be a padding row
        builder
            .when_transition()
            .when(local_is_padding_row.clone())
            .assert_one(next_is_padding_row.clone());
        // If we are in a digest row, the next row cannot be a digest row
        builder
            .when(*local_cols.flags.is_digest_row)
            .assert_zero(*next_cols.flags.is_digest_row);
        // Constrain how much the row index changes by
        // round->round: 1
        // round->digest: 1
        // digest->round: -C::ROUND_ROWS
        // digest->padding: 1
        // padding->padding: 0
        // Other transitions are not allowed by the above constraints
        let delta = *local_cols.flags.is_round_row * AB::Expr::ONE
            + *local_cols.flags.is_digest_row
                * *next_cols.flags.is_round_row
                * AB::Expr::from_canonical_usize(C::ROUND_ROWS)
                * AB::Expr::NEG_ONE
            + *local_cols.flags.is_digest_row * next_is_padding_row.clone() * AB::Expr::ONE;

        let local_row_idx = self.row_idx_encoder.flag_with_val::<AB>(
            local_cols.flags.row_idx.to_slice().unwrap(),
            &(0..=C::ROWS_PER_BLOCK).map(|i| (i, i)).collect::<Vec<_>>(),
        );
        let next_row_idx = self.row_idx_encoder.flag_with_val::<AB>(
            next_cols.flags.row_idx.to_slice().unwrap(),
            &(0..=C::ROWS_PER_BLOCK).map(|i| (i, i)).collect::<Vec<_>>(),
        );

        builder
            .when_transition()
            .assert_eq(local_row_idx.clone() + delta, next_row_idx.clone());
        builder.when_first_row().assert_zero(local_row_idx);

        // Constrain the global block index
        // We set the global block index to 0 for padding rows
        // Starting with 1 so it is not the same as the padding rows

        // Global block index is 1 on first row
        builder
            .when_first_row()
            .assert_one(*local_cols.flags.global_block_idx);

        // Global block index is constant on all rows in a block
        builder.when(*local_cols.flags.is_round_row).assert_eq(
            *local_cols.flags.global_block_idx,
            *next_cols.flags.global_block_idx,
        );
        // Global block index increases by 1 between blocks
        builder
            .when_transition()
            .when(*local_cols.flags.is_digest_row)
            .when(*next_cols.flags.is_round_row)
            .assert_eq(
                *local_cols.flags.global_block_idx + AB::Expr::ONE,
                *next_cols.flags.global_block_idx,
            );
        // Global block index is 0 on padding rows
        builder
            .when(local_is_padding_row.clone())
            .assert_zero(*local_cols.flags.global_block_idx);

        // Constrain the local block index
        // We set the local block index to 0 for padding rows

        // Local block index is constant on all rows in a block
        // and its value on padding rows is equal to its value on the first block
        builder
            .when(not(*local_cols.flags.is_digest_row))
            .assert_eq(
                *local_cols.flags.local_block_idx,
                *next_cols.flags.local_block_idx,
            );
        // Local block index increases by 1 between blocks in the same message
        builder
            .when(*local_cols.flags.is_digest_row)
            .when(not(*local_cols.flags.is_last_block))
            .assert_eq(
                *local_cols.flags.local_block_idx + AB::Expr::ONE,
                *next_cols.flags.local_block_idx,
            );
        // Local block index is 0 on padding rows
        // Combined with the above, this means that the local block index is 0 in the first block
        builder
            .when(*local_cols.flags.is_digest_row)
            .when(*local_cols.flags.is_last_block)
            .assert_zero(*next_cols.flags.local_block_idx);

        self.eval_message_schedule(builder, local_cols.clone(), next_cols.clone());
        self.eval_work_vars(builder, local_cols.clone(), next_cols);
        let next_cols: Sha2DigestColsRef<AB::Var> =
            Sha2DigestColsRef::from::<C>(&next[..C::DIGEST_WIDTH]);
        self.eval_digest_row(builder, local_cols, next_cols);
        let local_cols: Sha2DigestColsRef<AB::Var> =
            Sha2DigestColsRef::from::<C>(&local[..C::DIGEST_WIDTH]);
        self.eval_prev_hash(builder, local_cols, next_is_padding_row);
    }

    /// Constrains that the next block's `prev_hash` is equal to the current block's `hash`
    /// Note: the constraining is done by interactions with the chip itself on every digest row
    fn eval_prev_hash<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: Sha2DigestColsRef<AB::Var>,
        is_last_block_of_trace: AB::Expr, /* note this indicates the last block of the trace,
                                           * not the last block of the message */
    ) {
        // Constrain that next block's `prev_hash` is equal to the current block's `hash`
        let composed_hash = (0..C::HASH_WORDS)
            .map(|i| {
                let hash_bits = if i < C::ROUNDS_PER_ROW {
                    local
                        .hash
                        .a
                        .row(C::ROUNDS_PER_ROW - 1 - i)
                        .mapv(|x| x.into())
                        .to_vec()
                } else {
                    local
                        .hash
                        .e
                        .row(C::ROUNDS_PER_ROW + 3 - i)
                        .mapv(|x| x.into())
                        .to_vec()
                };
                (0..C::WORD_U16S)
                    .map(|j| compose::<AB::Expr>(&hash_bits[j * 16..(j + 1) * 16], 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Need to handle the case if this is the very last block of the trace matrix
        let next_global_block_idx = select(
            is_last_block_of_trace,
            AB::Expr::ONE,
            *local.flags.global_block_idx + AB::Expr::ONE,
        );
        // The following interactions constrain certain values from block to block
        self.bus.send(
            builder,
            composed_hash
                .into_iter()
                .flatten()
                .chain(once(next_global_block_idx)),
            *local.flags.is_digest_row,
        );

        self.bus.receive(
            builder,
            local
                .prev_hash
                .flatten()
                .mapv(|x| x.into())
                .into_iter()
                .chain(once((*local.flags.global_block_idx).into())),
            *local.flags.is_digest_row,
        );
    }

    /// Constrain the message schedule additions for `next` row
    /// Note: For every addition we need to constrain the following for each of [WORD_U16S] limbs
    /// sig_1(w_{t-2})[i] + w_{t-7}[i] + sig_0(w_{t-15})[i] + w_{t-16}[i] + carry_w[t][i-1] -
    /// carry_w[t][i] * 2^16 - w_t[i] == 0 Refer to [https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf]
    fn eval_message_schedule<'a, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: Sha2RoundColsRef<'a, AB::Var>,
        next: Sha2RoundColsRef<'a, AB::Var>,
    ) {
        // This `w` array contains 8 message schedule words - w_{idx}, ..., w_{idx+7} for some idx
        let w = ndarray::concatenate(
            ndarray::Axis(0),
            &[local.message_schedule.w, next.message_schedule.w],
        )
        .unwrap();

        // Constrain `w_3` for `next` row
        for i in 0..C::ROUNDS_PER_ROW - 1 {
            // here we constrain the w_3 of the i_th word of the next row
            // w_3 of next is w[i+4-3] = w[i+1]
            let w_3 = w.row(i + 1).mapv(|x| x.into()).to_vec();
            let expected_w_3 = next.schedule_helper.w_3.row(i);
            for j in 0..C::WORD_U16S {
                let w_3_limb = compose::<AB::Expr>(&w_3[j * 16..(j + 1) * 16], 1);
                builder
                    .when(*local.flags.is_round_row)
                    .assert_eq(w_3_limb, expected_w_3[j].into());
            }
        }

        // Constrain intermed for `next` row
        // We will only constrain intermed_12 for rows [3, C::ROUND_ROWS - 2], and let it
        // unconstrained for other rows Other rows should put the needed value in
        // intermed_12 to make the below summation constraint hold
        let is_row_intermed_12 = self.row_idx_encoder.contains_flag_range::<AB>(
            next.flags.row_idx.to_slice().unwrap(),
            3..=C::ROUND_ROWS - 2,
        );
        // We will only constrain intermed_8 for rows [2, C::ROUND_ROWS - 3], and let it
        // unconstrained for other rows
        let is_row_intermed_8 = self.row_idx_encoder.contains_flag_range::<AB>(
            next.flags.row_idx.to_slice().unwrap(),
            2..=C::ROUND_ROWS - 3,
        );
        for i in 0..C::ROUNDS_PER_ROW {
            // w_idx
            let w_idx = w.row(i).mapv(|x| x.into()).to_vec();
            // sig_0(w_{idx+1})
            let sig_w = small_sig0_field::<AB::Expr, C>(w.row(i + 1).as_slice().unwrap());
            for j in 0..C::WORD_U16S {
                let w_idx_limb = compose::<AB::Expr>(&w_idx[j * 16..(j + 1) * 16], 1);
                let sig_w_limb = compose::<AB::Expr>(&sig_w[j * 16..(j + 1) * 16], 1);

                // We would like to constrain this only on rows 0..16, but we can't do a conditional
                // check because the degree is already 3. So we must fill in
                // `intermed_4` with dummy values on rows 0 and 16 to ensure the constraint holds on
                // these rows.
                builder.when_transition().assert_eq(
                    next.schedule_helper.intermed_4[[i, j]],
                    w_idx_limb + sig_w_limb,
                );

                builder.when(is_row_intermed_8.clone()).assert_eq(
                    next.schedule_helper.intermed_8[[i, j]],
                    local.schedule_helper.intermed_4[[i, j]],
                );

                builder.when(is_row_intermed_12.clone()).assert_eq(
                    next.schedule_helper.intermed_12[[i, j]],
                    local.schedule_helper.intermed_8[[i, j]],
                );
            }
        }

        // Constrain the message schedule additions for `next` row
        for i in 0..C::ROUNDS_PER_ROW {
            // Note, here by w_{t} we mean the i_th word of the `next` row
            // w_{t-7}
            let w_7 = if i < 3 {
                local.schedule_helper.w_3.row(i).mapv(|x| x.into()).to_vec()
            } else {
                let w_3 = w.row(i - 3).mapv(|x| x.into()).to_vec();
                (0..C::WORD_U16S)
                    .map(|j| compose::<AB::Expr>(&w_3[j * 16..(j + 1) * 16], 1))
                    .collect::<Vec<_>>()
            };
            // sig_0(w_{t-15}) + w_{t-16}
            let intermed_16 = local.schedule_helper.intermed_12.row(i).mapv(|x| x.into());

            let carries = (0..C::WORD_U16S)
                .map(|j| {
                    next.message_schedule.carry_or_buffer[[i, j * 2]]
                        + AB::Expr::TWO * next.message_schedule.carry_or_buffer[[i, j * 2 + 1]]
                })
                .collect::<Vec<_>>();

            // Constrain `W_{idx} = sig_1(W_{idx-2}) + W_{idx-7} + sig_0(W_{idx-15}) + W_{idx-16}`
            // We would like to constrain this only on rows 4..C::ROUND_ROWS, but we can't do a
            // conditional check because the degree of sum is already 3 So we must fill
            // in `intermed_12` with dummy values on rows 0..3 and C::ROUND_ROWS-1 and C::ROUND_ROWS
            // to ensure the constraint holds on rows 0..4 and C::ROUND_ROWS. Note that
            // the dummy value goes in the previous row to make the current row's constraint hold.
            constraint_word_addition::<_, C>(
                // Note: here we can't do a conditional check because the degree of sum is already
                // 3
                &mut builder.when_transition(),
                &[&small_sig1_field::<AB::Expr, C>(
                    w.row(i + 2).as_slice().unwrap(),
                )],
                &[&w_7, intermed_16.as_slice().unwrap()],
                w.row(i + 4).as_slice().unwrap(),
                &carries,
            );

            for j in 0..C::WORD_U16S {
                // When on rows 4..C::ROUND_ROWS message schedule carries should be 0 or 1
                let is_row_4_or_more = *next.flags.is_round_row - *next.flags.is_first_4_rows;
                builder
                    .when(is_row_4_or_more.clone())
                    .assert_bool(next.message_schedule.carry_or_buffer[[i, j * 2]]);
                builder
                    .when(is_row_4_or_more)
                    .assert_bool(next.message_schedule.carry_or_buffer[[i, j * 2 + 1]]);
            }
            // Constrain w being composed of bits
            for j in 0..C::WORD_BITS {
                builder
                    .when(*next.flags.is_round_row)
                    .assert_bool(next.message_schedule.w[[i, j]]);
            }
        }
    }

    /// Constrain the work vars on `next` row according to the sha documentation
    /// Refer to [https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf]
    fn eval_work_vars<'a, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: Sha2RoundColsRef<'a, AB::Var>,
        next: Sha2RoundColsRef<'a, AB::Var>,
    ) {
        let a =
            ndarray::concatenate(ndarray::Axis(0), &[local.work_vars.a, next.work_vars.a]).unwrap();
        let e =
            ndarray::concatenate(ndarray::Axis(0), &[local.work_vars.e, next.work_vars.e]).unwrap();

        for i in 0..C::ROUNDS_PER_ROW {
            for j in 0..C::WORD_U16S {
                // Although we need carry_a <= 6 and carry_e <= 5, constraining carry_a, carry_e in
                // [0, 2^8) is enough to prevent overflow and ensure the soundness
                // of the addition we want to check
                self.bitwise_lookup_bus
                    .send_range(
                        local.work_vars.carry_a[[i, j]],
                        local.work_vars.carry_e[[i, j]],
                    )
                    .eval(builder, *local.flags.is_round_row);
            }

            let w_limbs = (0..C::WORD_U16S)
                .map(|j| {
                    compose::<AB::Expr>(
                        next.message_schedule
                            .w
                            .slice(s![i, j * 16..(j + 1) * 16])
                            .as_slice()
                            .unwrap(),
                        1,
                    ) * *next.flags.is_round_row
                })
                .collect::<Vec<_>>();

            let k_limbs = (0..C::WORD_U16S)
                .map(|j| {
                    self.row_idx_encoder.flag_with_val::<AB>(
                        next.flags.row_idx.to_slice().unwrap(),
                        &(0..C::ROUND_ROWS)
                            .map(|rw_idx| {
                                (
                                    rw_idx,
                                    word_into_u16_limbs::<C>(
                                        C::get_k()[rw_idx * C::ROUNDS_PER_ROW + i],
                                    )[j] as usize,
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();

            // Constrain `a = h + sig_1(e) + ch(e, f, g) + K + W + sig_0(a) + Maj(a, b, c)`
            // We have to enforce this constraint on all rows since the degree of the constraint is
            // already 3. So, we must fill in `carry_a` with dummy values on digest rows
            // to ensure the constraint holds.
            constraint_word_addition::<_, C>(
                builder,
                &[
                    e.row(i).mapv(|x| x.into()).as_slice().unwrap(), // previous `h`
                    &big_sig1_field::<AB::Expr, C>(e.row(i + 3).as_slice().unwrap()), /* sig_1 of previous `e` */
                    &ch_field::<AB::Expr>(
                        e.row(i + 3).as_slice().unwrap(),
                        e.row(i + 2).as_slice().unwrap(),
                        e.row(i + 1).as_slice().unwrap(),
                    ), /* Ch of previous `e`, `f`, `g` */
                    &big_sig0_field::<AB::Expr, C>(a.row(i + 3).as_slice().unwrap()), /* sig_0 of previous `a` */
                    &maj_field::<AB::Expr>(
                        a.row(i + 3).as_slice().unwrap(),
                        a.row(i + 2).as_slice().unwrap(),
                        a.row(i + 1).as_slice().unwrap(),
                    ), /* Maj of previous a, b, c */
                ],
                &[&w_limbs, &k_limbs],                             // K and W
                a.row(i + 4).as_slice().unwrap(),                  // new `a`
                next.work_vars.carry_a.row(i).as_slice().unwrap(), // carries of addition
            );

            // Constrain `e = d + h + sig_1(e) + ch(e, f, g) + K + W`
            // We have to enforce this constraint on all rows since the degree of the constraint is
            // already 3. So, we must fill in `carry_e` with dummy values on digest rows
            // to ensure the constraint holds.
            constraint_word_addition::<_, C>(
                builder,
                &[
                    a.row(i).mapv(|x| x.into()).as_slice().unwrap(), // previous `d`
                    e.row(i).mapv(|x| x.into()).as_slice().unwrap(), // previous `h`
                    &big_sig1_field::<AB::Expr, C>(e.row(i + 3).as_slice().unwrap()), /* sig_1 of previous `e` */
                    &ch_field::<AB::Expr>(
                        e.row(i + 3).as_slice().unwrap(),
                        e.row(i + 2).as_slice().unwrap(),
                        e.row(i + 1).as_slice().unwrap(),
                    ), /* Ch of previous `e`, `f`, `g` */
                ],
                &[&w_limbs, &k_limbs],                             // K and W
                e.row(i + 4).as_slice().unwrap(),                  // new `e`
                next.work_vars.carry_e.row(i).as_slice().unwrap(), // carries of addition
            );
        }
    }
}
