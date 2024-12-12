use std::{array, borrow::Borrow, cmp::max, iter::once};

use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    encoder::Encoder,
    utils::{not, select},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::AbstractField,
    p3_matrix::Matrix,
};

use super::{
    big_sig0_field, big_sig1_field, ch_field, compose, contains_flag, contains_flag_range,
    flag_with_val, maj_field, small_sig0_field, small_sig1_field, u32_into_limbs, Sha256DigestCols,
    Sha256RoundCols, SHA256_DIGEST_WIDTH, SHA256_H, SHA256_HASH_WORDS, SHA256_K,
    SHA256_ROUNDS_PER_ROW, SHA256_ROUND_WIDTH, SHA256_WORD_BITS, SHA256_WORD_U16S, SHA256_WORD_U8S,
};

#[derive(Clone, Debug)]
pub struct Sha256Air {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub row_idx_encoder: Encoder,
    bus_idx: usize,
}

impl Sha256Air {
    pub fn new(bitwise_lookup_bus: BitwiseOperationLookupBus, self_bus_idx: usize) -> Self {
        Self {
            bitwise_lookup_bus,
            row_idx_encoder: Encoder::new(17, 2, false),
            bus_idx: self_bus_idx,
        }
    }
}

impl<F> BaseAir<F> for Sha256Air {
    fn width(&self) -> usize {
        max(
            Sha256RoundCols::<F>::width(),
            Sha256DigestCols::<F>::width(),
        )
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for Sha256Air {
    /// The start column for the sub-air to use
    type AirContext<'a>
        = usize
    where
        Self: 'a,
        AB: 'a,
        <AB as AirBuilder>::Var: 'a,
        <AB as AirBuilder>::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, start_col: Self::AirContext<'a>)
    where
        <AB as AirBuilder>::Var: 'a,
        <AB as AirBuilder>::Expr: 'a,
    {
        self.eval_row(builder, start_col);
        self.eval_transitions(builder, start_col);
    }
}

impl Sha256Air {
    /// Implement constraints that need to be true for all rows
    fn eval_row<AB: InteractionBuilder>(&self, builder: &mut AB, start_col: usize) {
        let main = builder.main();
        let local = main.row_slice(0);

        // Doesn't matter which column struct we use here
        let local_cols: &Sha256RoundCols<AB::Var> =
            local[start_col..start_col + SHA256_ROUND_WIDTH].borrow();
        let flags = &local_cols.flags;
        builder.assert_bool(flags.is_round_row);
        builder.assert_bool(flags.is_first_4_rows);
        builder.assert_bool(flags.is_digest_row);
        builder.assert_bool(flags.is_round_row + flags.is_digest_row);
        builder.assert_bool(flags.is_last_block);

        self.row_idx_encoder
            .eval(builder, &local_cols.flags.row_idx);
        builder.assert_one(contains_flag_range::<AB>(
            &self.row_idx_encoder,
            &local_cols.flags.row_idx,
            0,
            17,
        ));
        builder.assert_eq(
            contains_flag_range::<AB>(&self.row_idx_encoder, &local_cols.flags.row_idx, 0, 3),
            flags.is_first_4_rows,
        );
        builder.assert_eq(
            contains_flag_range::<AB>(&self.row_idx_encoder, &local_cols.flags.row_idx, 0, 15),
            flags.is_round_row,
        );
        builder.assert_eq(
            contains_flag::<AB>(&self.row_idx_encoder, &local_cols.flags.row_idx, &[16]),
            flags.is_digest_row,
        );
        // If invalid row we want the row_idx to be 17
        builder.assert_eq(
            contains_flag::<AB>(&self.row_idx_encoder, &local_cols.flags.row_idx, &[17]),
            not::<AB::Expr>(flags.is_digest_row + flags.is_round_row),
        );

        // Constrain a, e, being composed of bits: we make sure a and e are always in the same place in the trace matrix
        // Note: this has to be true for every row, even invalid rows
        for i in 0..SHA256_ROUNDS_PER_ROW {
            for j in 0..SHA256_WORD_BITS {
                builder.assert_bool(local_cols.work_vars.a[i][j]);
                builder.assert_bool(local_cols.work_vars.e[i][j]);
            }
        }
        self.eval_round_row(builder, local_cols);
        let local_cols: &Sha256DigestCols<AB::Var> =
            local[start_col..start_col + SHA256_DIGEST_WIDTH].borrow();
        self.eval_digest_row(builder, local_cols);
    }

    /// Implement constraints for a row as if it is a round row
    fn eval_round_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha256RoundCols<AB::Var>,
    ) {
        for i in 0..SHA256_ROUNDS_PER_ROW {
            // Constrain w being composed of bits
            for j in 0..SHA256_WORD_BITS {
                builder
                    .when(local.flags.is_round_row)
                    .assert_bool(local.message_schedule.w[i][j]);
            }
            for j in 0..SHA256_WORD_U16S {
                // Although we need carry_a <= 6 and carry_e <= 5, constraining carry_a, carry_e in [0, 2^8) is enough
                // to prevent overflow and ensure the soundness of the addition we want to check
                self.bitwise_lookup_bus
                    .send_range(local.work_vars.carry_a[i][j], local.work_vars.carry_e[i][j])
                    .eval(builder, local.flags.is_round_row);

                // When on rows 4..16 message schedule carries should be 0 or 1
                let is_row_4_15 = local.flags.is_round_row - local.flags.is_first_4_rows;
                builder
                    .when(is_row_4_15.clone())
                    .assert_bool(local.message_schedule.carry_or_buffer[i][j * 2]);
                builder
                    .when(is_row_4_15)
                    .assert_bool(local.message_schedule.carry_or_buffer[i][j * 2 + 1]);
            }
        }
    }

    /// Implement constraints for a digest row
    fn eval_digest_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha256DigestCols<AB::Var>,
    ) {
        // Check that if this is the last row of a message or an invalid row, the hash should be the [SHA256_H]
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let a = local.hash.a[i].map(|x| x.into());
            let e = local.hash.e[i].map(|x| x.into());
            for j in 0..SHA256_WORD_U16S {
                let a_limb = compose::<AB::Expr>(&a[j * 16..(j + 1) * 16], 1);
                let e_limb = compose::<AB::Expr>(&e[j * 16..(j + 1) * 16], 1);

                // If it is a padding row or the last row of a message, the `hash` should be the [SHA256_H]
                builder
                    .when(
                        not::<AB::Expr>(local.flags.is_round_row + local.flags.is_digest_row)
                            + local.flags.is_last_block * local.flags.is_digest_row,
                    )
                    .assert_eq(
                        a_limb,
                        AB::Expr::from_canonical_u32(
                            u32_into_limbs::<2>(SHA256_H[SHA256_ROUNDS_PER_ROW - i - 1])[j],
                        ),
                    );

                builder
                    .when(
                        not::<AB::Expr>(local.flags.is_round_row + local.flags.is_digest_row)
                            + local.flags.is_last_block * local.flags.is_digest_row,
                    )
                    .assert_eq(
                        e_limb,
                        AB::Expr::from_canonical_u32(
                            u32_into_limbs::<2>(SHA256_H[SHA256_ROUNDS_PER_ROW - i + 3])[j],
                        ),
                    );
            }
        }

        // Check if last row of a non-last block, the `hash` should be equal to the final hash of the current block
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let prev_a = local.hash.a[i].map(|x| x.into());
            let prev_e = local.hash.e[i].map(|x| x.into());
            let cur_a = local.final_hash[SHA256_ROUNDS_PER_ROW - i - 1].map(|x| x.into());
            let cur_e = local.final_hash[SHA256_ROUNDS_PER_ROW - i + 3].map(|x| x.into());
            for j in 0..SHA256_WORD_U8S {
                let prev_a_limb = compose::<AB::Expr>(&prev_a[j * 8..(j + 1) * 8], 1);
                let prev_e_limb = compose::<AB::Expr>(&prev_e[j * 8..(j + 1) * 8], 1);

                builder
                    .when(not(local.flags.is_last_block) * local.flags.is_digest_row)
                    .assert_eq(prev_a_limb, cur_a[j].clone());

                builder
                    .when(not(local.flags.is_last_block) * local.flags.is_digest_row)
                    .assert_eq(prev_e_limb, cur_e[j].clone());
            }
        }
    }

    fn eval_transitions<AB: InteractionBuilder>(&self, builder: &mut AB, start_col: usize) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // Doesn't matter what column structs we use here
        let local_cols: &Sha256RoundCols<AB::Var> =
            local[start_col..start_col + SHA256_ROUND_WIDTH].borrow();
        let next_cols: &Sha256RoundCols<AB::Var> =
            next[start_col..start_col + SHA256_ROUND_WIDTH].borrow();

        let local_is_padding_row =
            not::<AB::Expr>(local_cols.flags.is_round_row + local_cols.flags.is_digest_row);
        let next_is_padding_row =
            not::<AB::Expr>(next_cols.flags.is_round_row + next_cols.flags.is_digest_row);
        // Checking the very last block has `is_last_block` -> at least one block is a `is_last_block`
        // the rest of the constraining of `is_last_block` should be done by the wrapper chip
        builder
            .when(next_is_padding_row.clone())
            .when(local_cols.flags.is_digest_row)
            .assert_one(local_cols.flags.is_last_block);
        builder
            .when_last_row()
            .when(local_cols.flags.is_digest_row)
            .assert_one(local_cols.flags.is_last_block);
        // If we are in a round row, the next row cannot be a padding row
        builder
            .when(local_cols.flags.is_round_row)
            .assert_zero(next_is_padding_row.clone());
        // The first row must be a round row
        builder
            .when_first_row()
            .assert_one(local_cols.flags.is_round_row);
        // If we are in a padding row, the next row must also be a padding row
        builder
            .when_transition()
            .when(local_is_padding_row.clone())
            .assert_one(next_is_padding_row.clone());
        // If we are in a digest row, the next row cannot be a digest row
        builder
            .when(local_cols.flags.is_digest_row)
            .assert_zero(next_cols.flags.is_digest_row);
        // Constrin how much the row index changes by
        // round->round: 1
        // round->digest: 1
        // digest->round: -16
        // digest->padding: 0
        // padding->padding: 0
        // Other transitions are not allowed by the above
        let delta = local_cols.flags.is_round_row
            * (next_cols.flags.is_digest_row + next_cols.flags.is_round_row)
            * AB::Expr::ONE
            + local_cols.flags.is_digest_row
                * next_cols.flags.is_round_row
                * AB::Expr::from_canonical_u32(16)
                * AB::Expr::NEG_ONE
            + local_cols.flags.is_digest_row * next_is_padding_row.clone() * AB::Expr::ONE
            + local_is_padding_row.clone() * AB::Expr::ZERO;

        let local_row_idx = flag_with_val::<AB>(
            &self.row_idx_encoder,
            &local_cols.flags.row_idx,
            &(0..18).map(|i| (i, i)).collect::<Vec<_>>(),
        );
        let next_row_idx = flag_with_val::<AB>(
            &self.row_idx_encoder,
            &next_cols.flags.row_idx,
            &(0..18).map(|i| (i, i)).collect::<Vec<_>>(),
        );

        builder
            .when_transition()
            .assert_eq(local_row_idx.clone() + delta, next_row_idx.clone());
        builder.when_first_row().assert_zero(local_row_idx);

        // Constrain the global block index
        // We set the global block index to 0 for padding rows
        // Starting with 1 so it is not the same as the padding rows
        builder
            .when_first_row()
            .assert_one(local_cols.flags.global_block_idx);

        builder.when(local_cols.flags.is_round_row).assert_eq(
            local_cols.flags.global_block_idx,
            next_cols.flags.global_block_idx,
        );
        builder
            .when_transition()
            .when(local_cols.flags.is_digest_row)
            .when(next_cols.flags.is_round_row)
            .assert_eq(
                local_cols.flags.global_block_idx + AB::Expr::ONE,
                next_cols.flags.global_block_idx,
            );
        builder
            .when(local_is_padding_row.clone())
            .assert_zero(local_cols.flags.global_block_idx);

        // Constrain the local block index
        // We set the local block index to 0 for padding rows
        builder.when(not(local_cols.flags.is_digest_row)).assert_eq(
            local_cols.flags.local_block_idx,
            next_cols.flags.local_block_idx,
        );
        builder
            .when(local_cols.flags.is_digest_row)
            .when(not(local_cols.flags.is_last_block))
            .assert_eq(
                local_cols.flags.local_block_idx + AB::Expr::ONE,
                next_cols.flags.local_block_idx,
            );

        builder
            .when(local_cols.flags.is_digest_row)
            .when(local_cols.flags.is_last_block)
            .assert_zero(next_cols.flags.local_block_idx);

        self.eval_message_schedule::<AB>(builder, local_cols, next_cols);
        self.eval_work_vars::<AB>(builder, local_cols, next_cols);
        let local_cols: &Sha256DigestCols<AB::Var> =
            local[start_col..start_col + SHA256_DIGEST_WIDTH].borrow();
        self.eval_prev_hash::<AB>(builder, local_cols, next_is_padding_row);
    }

    /// Constrains that the next block's `prev_hash` is equal to the current block's `hash`
    /// Note: the constraining is done by interactions with the chip itself on every digest row
    fn eval_prev_hash<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha256DigestCols<AB::Var>,
        is_lastest_block: AB::Expr,
    ) {
        // Constrain that next block's `prev_hash` is equal to the current block's `hash`
        let composed_hash: [[<AB as AirBuilder>::Expr; SHA256_WORD_U16S]; SHA256_HASH_WORDS] =
            array::from_fn(|i| {
                let hash_bits = if i < SHA256_ROUNDS_PER_ROW {
                    local.hash.a[SHA256_ROUNDS_PER_ROW - 1 - i].map(|x| x.into())
                } else {
                    local.hash.e[SHA256_ROUNDS_PER_ROW + 3 - i].map(|x| x.into())
                };
                array::from_fn(|j| compose::<AB::Expr>(&hash_bits[j * 16..(j + 1) * 16], 1))
            });
        // Need to handle the case if this is the very last block of the trace matrix
        let next_global_block_idx = select(
            is_lastest_block,
            AB::Expr::ONE,
            local.flags.global_block_idx + AB::Expr::ONE,
        );
        // The following interactions constrain certain values from block to block
        builder.push_send(
            self.bus_idx,
            composed_hash
                .into_iter()
                .flatten()
                .chain(once(next_global_block_idx)),
            local.flags.is_digest_row,
        );

        builder.push_receive(
            self.bus_idx,
            local
                .prev_hash
                .into_iter()
                .flatten()
                .map(|x| x.into())
                .chain(once(local.flags.global_block_idx.into())),
            local.flags.is_digest_row,
        );
    }

    /// Constrain the message schedule additions
    /// Note: For every addition we need to constrain the following for each of [SHA256_WORD_U16S] limbs
    /// sig_1(w_{t-2})[i] + w_{t-7}[i] + sig_0(w_{t-15})[i] + w_{t-16}[i] + carry_w[t][i-1] - carry_w[t][i] * 2^16 - w_t[i] == 0
    fn eval_message_schedule<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha256RoundCols<AB::Var>,
        next: &Sha256RoundCols<AB::Var>,
    ) {
        let w = [local.message_schedule.w, next.message_schedule.w].concat();

        // Constrain `w_3`
        for i in 0..SHA256_ROUNDS_PER_ROW - 1 {
            let w_3 = w[i + 1].map(|x| x.into());
            let expected_w_3 = next.schedule_helper.w_3[i];
            for j in 0..SHA256_WORD_U16S {
                let w_3_limb = compose::<AB::Expr>(&w_3[j * 16..(j + 1) * 16], 1);
                builder.assert_eq(w_3_limb, expected_w_3[j].into());
            }
        }

        // Constrain intermed
        // We will only constrain intermed_12 for rows [3, 14], and let it unconstrained for other rows
        // Other rows should put the needed value in intermed_12 to make the below summation constraint hold
        let is_row_3_14 =
            contains_flag_range::<AB>(&self.row_idx_encoder, &next.flags.row_idx, 3, 14);
        for i in 0..SHA256_ROUNDS_PER_ROW {
            // w_idx
            let w_idx = w[i].map(|x| x.into());
            // sig_0(w_{idx+1})
            let sig_w = small_sig0_field::<AB::Expr>(&w[i + 1]);
            for j in 0..SHA256_WORD_U16S {
                let w_idx_limb = compose::<AB::Expr>(&w_idx[j * 16..(j + 1) * 16], 1);
                let sig_w_limb = compose::<AB::Expr>(&sig_w[j * 16..(j + 1) * 16], 1);

                builder.assert_eq(
                    next.schedule_helper.intermed_4[i][j],
                    w_idx_limb + sig_w_limb,
                );

                builder.assert_eq(
                    next.schedule_helper.intermed_8[i][j],
                    local.schedule_helper.intermed_4[i][j],
                );

                builder.when(is_row_3_14.clone()).assert_eq(
                    next.schedule_helper.intermed_12[i][j],
                    local.schedule_helper.intermed_8[i][j],
                );
            }
        }

        // Constrain the message schedule additions
        for i in 0..SHA256_ROUNDS_PER_ROW {
            // sig_1(w_{t-2})
            let sig_w_2: [_; SHA256_WORD_U16S] = array::from_fn(|j| {
                compose::<AB::Expr>(
                    &small_sig1_field::<AB::Expr>(&w[i + 2])[j * 16..(j + 1) * 16],
                    1,
                )
            });
            // w_{t-7}
            let w_7 = if i < 3 {
                local.schedule_helper.w_3[i].map(|x| x.into())
            } else {
                let w_3 = w[i - 3].map(|x| x.into());
                array::from_fn(|j| compose::<AB::Expr>(&w_3[j * 16..(j + 1) * 16], 1))
            };
            // sig_0(w_{t-15}) + w_{t-16}
            let intermed_16 = local.schedule_helper.intermed_12[i].map(|x| x.into());
            // w_t
            let w_cur = w[i + 4].map(|x| x.into());
            let w_cur: [_; SHA256_WORD_U16S] =
                array::from_fn(|j| compose::<AB::Expr>(&w_cur[j * 16..(j + 1) * 16], 1));

            for j in 0..SHA256_WORD_U16S {
                let carry = next.message_schedule.carry_or_buffer[i][j * 2]
                    + AB::Expr::TWO * next.message_schedule.carry_or_buffer[i][j * 2 + 1];
                let sum = sig_w_2[j].clone() + w_7[j].clone() + intermed_16[j].clone()
                    - carry * AB::Expr::from_canonical_u32(1 << 16)
                    - w_cur[j].clone()
                    + if j > 0 {
                        next.message_schedule.carry_or_buffer[i][j * 2 - 2]
                            + AB::Expr::TWO * next.message_schedule.carry_or_buffer[i][j * 2 - 1]
                    } else {
                        AB::Expr::ZERO
                    };
                // Note: here we can't do a conditional check because the degree of sum is already 3
                builder.assert_zero(sum);
            }
        }
    }

    /// Constrain the work vars on `next` row according to the sha256 documentation
    fn eval_work_vars<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha256RoundCols<AB::Var>,
        next: &Sha256RoundCols<AB::Var>,
    ) {
        let a = [local.work_vars.a, next.work_vars.a].concat();
        let e = [local.work_vars.e, next.work_vars.e].concat();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let cur_a = a[i + 4].map(|x| x.into());
            let sig_a = big_sig0_field::<AB::Expr>(&a[i + 3]);
            let maj_abc = maj_field::<AB::Expr>(&a[i + 3], &a[i + 2], &a[i + 1]);
            let d = a[i].map(|x| x.into());
            let cur_e = e[i + 4].map(|x| x.into());
            let sig_e = big_sig1_field::<AB::Expr>(&e[i + 3]);
            let ch_efg = ch_field::<AB::Expr>(&e[i + 3], &e[i + 2], &e[i + 1]);
            let h = e[i].map(|x| x.into());
            let w = next.message_schedule.w[i].map(|x| x.into());

            // k and w are not included in t1 here and are handled a bit differently
            let t1 = [h, sig_e, ch_efg];
            let t2 = [sig_a, maj_abc];
            for j in 0..SHA256_WORD_U16S {
                let w_limb =
                    compose::<AB::Expr>(&w[j * 16..(j + 1) * 16], 1) * next.flags.is_round_row;
                let k_limb = flag_with_val::<AB>(
                    &self.row_idx_encoder,
                    &next.flags.row_idx,
                    &(0..16)
                        .map(|rw_idx| {
                            (
                                rw_idx,
                                u32_into_limbs::<SHA256_WORD_U16S>(
                                    SHA256_K[rw_idx * SHA256_ROUNDS_PER_ROW + i],
                                )[j] as usize,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                let t1_limb_sum = t1.iter().fold(AB::Expr::ZERO, |acc, x| {
                    acc + compose::<AB::Expr>(&x[j * 16..(j + 1) * 16], 1)
                }) + w_limb
                    + k_limb;
                let t2_limb_sum = t2.iter().fold(AB::Expr::ZERO, |acc, x| {
                    acc + compose::<AB::Expr>(&x[j * 16..(j + 1) * 16], 1)
                });
                let d_limb = compose::<AB::Expr>(&d[j * 16..(j + 1) * 16], 1);

                // Constrain `e`
                let cur_e_limb = compose::<AB::Expr>(&cur_e[j * 16..(j + 1) * 16], 1);
                builder.assert_eq(
                    d_limb
                        + t1_limb_sum.clone()
                        + if j == 0 {
                            AB::Expr::ZERO
                        } else {
                            next.work_vars.carry_e[i][j - 1].into()
                        },
                    cur_e_limb
                        + next.work_vars.carry_e[i][j] * AB::Expr::from_canonical_u32(1 << 16),
                );

                // Constrain `a`
                let cur_a_limb = compose::<AB::Expr>(&cur_a[j * 16..(j + 1) * 16], 1);
                builder.assert_eq(
                    t1_limb_sum
                        + t2_limb_sum
                        + if j == 0 {
                            AB::Expr::ZERO
                        } else {
                            next.work_vars.carry_a[i][j - 1].into()
                        },
                    cur_a_limb
                        + next.work_vars.carry_a[i][j] * AB::Expr::from_canonical_u32(1 << 16),
                );
            }
        }
    }
}
