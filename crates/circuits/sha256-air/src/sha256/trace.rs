use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    air::Sha256Air, big_sig0_field, big_sig1_field, ch_field, columns::Sha256RoundCols, compose,
    get_flag_pt_array, maj_field, small_sig0_field, small_sig1_field, SHA256_BLOCK_WORDS,
    SHA256_DIGEST_WIDTH, SHA256_HASH_WORDS, SHA256_ROUND_WIDTH, SHA256_WIDTH,
};
use crate::sha256::{
    big_sig0, big_sig1, ch, columns::Sha256DigestCols, limbs_into_u32, maj, small_sig0, small_sig1,
    u32_into_limbs, SHA256_H, SHA256_K, SHA256_ROUNDS_PER_ROW, SHA256_ROWS_PER_BLOCK,
    SHA256_WORD_BITS, SHA256_WORD_U16S, SHA256_WORD_U8S,
};

/// It is important to call the generate functions in the correct order:
/// default_rows should be initialized first
/// generate_intermed_4 should be called on every row before generate_intermed_8 is called
/// generate_intermed_12 should be called at the very end, when everything else is filled in
impl Sha256Air {
    /// This function takes the intput_massage (should be already padded), the previous hash,
    /// a flag indicating if it's the last block, the global block index, the local block index,
    /// and the buffer values that will be put in rows 0..4. Returns the trace of the block and the final hash
    #[allow(clippy::too_many_arguments)]
    pub fn generate_block_trace<F: PrimeField32>(
        &self,
        input: &[u32; SHA256_BLOCK_WORDS],
        bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<8>>,
        prev_hash: &[u32; SHA256_HASH_WORDS],
        is_last_block: bool,
        global_block_idx: u32,
        local_block_idx: u32,
        buffer_vals: &[[u32; SHA256_WORD_U16S * SHA256_ROUNDS_PER_ROW * 2]; 4],
    ) -> ([[F; SHA256_WIDTH]; 17], [u32; SHA256_HASH_WORDS]) {
        assert!(self.bitwise_lookup_bus == bitwise_lookup_chip.bus());
        if local_block_idx == 0 {
            debug_assert!(*prev_hash == SHA256_H);
        }
        let mut output = [[F::ZERO; SHA256_WIDTH]; SHA256_ROWS_PER_BLOCK];
        let mut message_schedule: Vec<u32> = input.to_vec();
        let mut work_vars = *prev_hash;
        let mut final_hash = [0u32; SHA256_HASH_WORDS];
        for (i, row) in output.iter_mut().enumerate() {
            // doing the 64 rounds in 16 rows
            if i < 16 {
                let cols: &mut Sha256RoundCols<F> = row[0..SHA256_ROUND_WIDTH].borrow_mut();

                cols.flags.is_round_row = F::ONE;
                cols.flags.is_first_4_rows = if i < 4 { F::ONE } else { F::ZERO };
                cols.flags.is_digest_row = F::ZERO;
                cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags.row_idx =
                    get_flag_pt_array(&self.row_idx_encoder, i).map(F::from_canonical_u32);
                cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);
                cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);

                // W_idx = M_idx
                if i < SHA256_ROWS_PER_BLOCK / SHA256_ROUNDS_PER_ROW {
                    for j in 0..SHA256_ROUNDS_PER_ROW {
                        cols.message_schedule.w[j] = u32_into_limbs::<SHA256_WORD_BITS>(
                            input[i * SHA256_ROUNDS_PER_ROW + j],
                        )
                        .map(F::from_canonical_u32);
                        cols.message_schedule.carry_or_buffer[j] = array::from_fn(|k| {
                            F::from_canonical_u32(buffer_vals[i][j * SHA256_WORD_U16S * 2 + k])
                        });
                    }
                }
                // W_idx = SIG1(W_{idx-2}) + W_{idx-7} + SIG0(W_{idx-15}) + W_{idx-16}
                else {
                    for j in 0..SHA256_ROUNDS_PER_ROW {
                        let idx = i * SHA256_ROUNDS_PER_ROW + j;
                        let nums: [u32; 4] = [
                            small_sig1(message_schedule[idx - 2]),
                            message_schedule[idx - 7],
                            small_sig0(message_schedule[idx - 15]),
                            message_schedule[idx - 16],
                        ];
                        let w: u32 = nums.iter().fold(0, |acc, &num| acc.wrapping_add(num));
                        cols.message_schedule.w[j] =
                            u32_into_limbs::<SHA256_WORD_BITS>(w).map(F::from_canonical_u32);

                        // fill in the carrys
                        for k in 0..SHA256_WORD_U16S {
                            let mut sum = nums.iter().fold(0, |acc, &num| {
                                acc + u32_into_limbs::<SHA256_WORD_U16S>(num)[k]
                            });
                            if k > 0 {
                                sum += cols.message_schedule.carry_or_buffer[j][k * 2 - 2]
                                    .as_canonical_u32()
                                    + 2 * cols.message_schedule.carry_or_buffer[j][k * 2 - 1]
                                        .as_canonical_u32();
                            }
                            let carry = (sum - u32_into_limbs::<SHA256_WORD_U16S>(w)[k]) >> 16;
                            cols.message_schedule.carry_or_buffer[j][k * 2] =
                                F::from_canonical_u32(carry & 1);
                            cols.message_schedule.carry_or_buffer[j][k * 2 + 1] =
                                F::from_canonical_u32(carry >> 1);
                        }
                        // update the message schedule
                        message_schedule.push(w);
                    }
                }
                // fill in the work variables
                for j in 0..SHA256_ROUNDS_PER_ROW {
                    // t1 = h + SIG1(e) + ch(e, f, g) + K_idx + W_idx
                    let t1 = [
                        work_vars[7],
                        big_sig1(work_vars[4]),
                        ch(work_vars[4], work_vars[5], work_vars[6]),
                        SHA256_K[i * SHA256_ROUNDS_PER_ROW + j],
                        limbs_into_u32(cols.message_schedule.w[j].map(|f| f.as_canonical_u32())),
                    ];
                    let t1_sum: u32 = t1.iter().fold(0, |acc, &num| acc.wrapping_add(num));

                    // t2 = SIG0(a) + maj(a, b, c)
                    let t2 = [
                        big_sig0(work_vars[0]),
                        maj(work_vars[0], work_vars[1], work_vars[2]),
                    ];

                    let t2_sum: u32 = t2.iter().fold(0, |acc, &num| acc.wrapping_add(num));

                    // e = d + t1
                    let e = work_vars[3].wrapping_add(t1_sum);
                    cols.work_vars.e[j] =
                        u32_into_limbs::<SHA256_WORD_BITS>(e).map(F::from_canonical_u32);
                    // a = t1 + t2
                    let a = t1_sum.wrapping_add(t2_sum);
                    cols.work_vars.a[j] =
                        u32_into_limbs::<SHA256_WORD_BITS>(a).map(F::from_canonical_u32);

                    // fill in the carrys
                    for k in 0..SHA256_WORD_U16S {
                        let t1_limb = t1.iter().fold(0, |acc, &num| {
                            acc + u32_into_limbs::<SHA256_WORD_U16S>(num)[k]
                        });
                        let t2_limb = t2.iter().fold(0, |acc, &num| {
                            acc + u32_into_limbs::<SHA256_WORD_U16S>(num)[k]
                        });

                        let mut e_limb =
                            t1_limb + u32_into_limbs::<SHA256_WORD_U16S>(work_vars[3])[k];
                        let mut a_limb = t1_limb + t2_limb;
                        if k > 0 {
                            a_limb += cols.work_vars.carry_a[j][k - 1].as_canonical_u32();
                            e_limb += cols.work_vars.carry_e[j][k - 1].as_canonical_u32();
                        }
                        let carry_a = (a_limb - u32_into_limbs::<SHA256_WORD_U16S>(a)[k]) >> 16;
                        let carry_e = (e_limb - u32_into_limbs::<SHA256_WORD_U16S>(e)[k]) >> 16;
                        cols.work_vars.carry_a[j][k] = F::from_canonical_u32(carry_a);
                        cols.work_vars.carry_e[j][k] = F::from_canonical_u32(carry_e);
                        bitwise_lookup_chip.request_range(carry_a, carry_e);
                    }

                    // update working variables
                    work_vars[7] = work_vars[6];
                    work_vars[6] = work_vars[5];
                    work_vars[5] = work_vars[4];
                    work_vars[4] = e;
                    work_vars[3] = work_vars[2];
                    work_vars[2] = work_vars[1];
                    work_vars[1] = work_vars[0];
                    work_vars[0] = a;
                }
            }
            // computing the final hash
            else {
                let cols: &mut Sha256DigestCols<F> = row[..SHA256_DIGEST_WIDTH].borrow_mut();
                cols.flags.is_round_row = F::ZERO;
                cols.flags.is_first_4_rows = F::ZERO;
                cols.flags.is_digest_row = F::ONE;
                cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags.row_idx =
                    get_flag_pt_array(&self.row_idx_encoder, 16).map(F::from_canonical_u32);
                cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);

                cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);
                final_hash = array::from_fn(|i| work_vars[i].wrapping_add(prev_hash[i]));
                cols.final_hash = array::from_fn(|i| {
                    u32_into_limbs::<SHA256_WORD_U8S>(final_hash[i]).map(F::from_canonical_u32)
                });
                cols.prev_hash = prev_hash
                    .map(|f| u32_into_limbs::<SHA256_WORD_U16S>(f).map(F::from_canonical_u32));
                let hash = if is_last_block {
                    SHA256_H.map(u32_into_limbs::<SHA256_WORD_BITS>)
                } else {
                    cols.final_hash
                        .map(|f| limbs_into_u32(f.map(|x| x.as_canonical_u32())))
                        .map(u32_into_limbs::<SHA256_WORD_BITS>)
                }
                .map(|x| x.map(F::from_canonical_u32));

                for i in 0..SHA256_ROUNDS_PER_ROW {
                    cols.hash.a[i] = hash[SHA256_ROUNDS_PER_ROW - i - 1];
                    cols.hash.e[i] = hash[SHA256_ROUNDS_PER_ROW - i + 3];
                }
            }
        }
        let (first16, last1) = output.split_at_mut(16);
        let local_cols: &Sha256RoundCols<F> = first16[15].as_slice().borrow();
        let next_cols: &mut Sha256RoundCols<F> = last1[0].as_mut_slice().borrow_mut();
        Self::generate_carry_ae(local_cols, next_cols, 16);
        (output, final_hash)
    }

    /// Puts the correct carrys in the `next_row`, the resulting carrys can be out of bound
    /// Here, row_idx is the index of the `next_row` inside the block: 0..16 for the first 16 rows and
    /// some value outside of the 0..16 range for other rows (possibly invalid)
    fn generate_carry_ae<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
        row_idx: usize,
    ) {
        let a = [local_cols.work_vars.a, next_cols.work_vars.a].concat();
        let e = [local_cols.work_vars.e, next_cols.work_vars.e].concat();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let cur_a = a[i + 4];
            let sig_a = big_sig0_field::<F>(&a[i + 3]);
            let maj_abc = maj_field::<F>(&a[i + 3], &a[i + 2], &a[i + 1]);
            let d = a[i];
            let cur_e = e[i + 4];
            let sig_e = big_sig1_field::<F>(&e[i + 3]);
            let ch_efg = ch_field::<F>(&e[i + 3], &e[i + 2], &e[i + 1]);
            let h = e[i];
            let w = if row_idx < 16 {
                next_cols.message_schedule.w[i]
            } else {
                [F::ZERO; 32]
            };

            let t1 = [h, sig_e, ch_efg, w];
            let t2 = [sig_a, maj_abc];
            for j in 0..SHA256_WORD_U16S {
                let t1_limb_sum = t1.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let t2_limb_sum = t2.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let d_limb = compose::<F>(&d[j * 16..(j + 1) * 16], 1);
                let k_limb = if row_idx < 16 {
                    F::from_canonical_u32(
                        u32_into_limbs::<SHA256_WORD_U16S>(
                            SHA256_K[row_idx * SHA256_ROUNDS_PER_ROW + i],
                        )[j],
                    )
                } else {
                    F::ZERO
                };
                let cur_a_limb = compose::<F>(&cur_a[j * 16..(j + 1) * 16], 1);
                let cur_e_limb = compose::<F>(&cur_e[j * 16..(j + 1) * 16], 1);
                let sum = d_limb
                    + t1_limb_sum
                    + k_limb
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_e[i][j - 1]
                    }
                    - cur_e_limb;
                let carry_e = sum * (F::from_canonical_u32(1 << 16).inverse());

                let sum = t1_limb_sum
                    + t2_limb_sum
                    + k_limb
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_a[i][j - 1]
                    }
                    - cur_a_limb;
                let carry_a = sum * (F::from_canonical_u32(1 << 16).inverse());
                next_cols.work_vars.carry_e[i][j] = carry_e;
                next_cols.work_vars.carry_a[i][j] = carry_a;
            }
        }
    }

    /// Puts the correct intermed_4 in the `next_row`
    pub fn generate_intermed_4<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        let w = [local_cols.message_schedule.w, next_cols.message_schedule.w].concat();
        let w_limbs: Vec<[F; SHA256_WORD_U16S]> = w
            .iter()
            .map(|x| array::from_fn(|i| compose::<F>(&x[i * 16..(i + 1) * 16], 1)))
            .collect();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let sig_w = small_sig0_field::<F>(&w[i + 1]);
            let sig_w_limbs: [F; SHA256_WORD_U16S] =
                array::from_fn(|j| compose::<F>(&sig_w[j * 16..(j + 1) * 16], 1));
            for (j, sig_w_limb) in sig_w_limbs.iter().enumerate() {
                next_cols.schedule_helper.intermed_4[i][j] = w_limbs[i][j] + *sig_w_limb;
            }
        }
    }

    /// Puts the correct intermed_8 in the `next_row`
    pub fn generate_intermed_8<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        for i in 0..SHA256_ROUNDS_PER_ROW {
            for j in 0..SHA256_WORD_U16S {
                next_cols.schedule_helper.intermed_8[i][j] =
                    local_cols.schedule_helper.intermed_4[i][j];
            }
        }
    }

    /// Puts the needed intermed_12 in the `local_row`
    pub fn generate_intermed_12<F: PrimeField32>(
        local_cols: &mut Sha256RoundCols<F>,
        next_cols: &Sha256RoundCols<F>,
    ) {
        let w = [local_cols.message_schedule.w, next_cols.message_schedule.w].concat();
        let w_limbs: Vec<[F; SHA256_WORD_U16S]> = w
            .iter()
            .map(|x| array::from_fn(|i| compose::<F>(&x[i * 16..(i + 1) * 16], 1)))
            .collect();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            // sig_1(w_{t-2})
            let sig_w_2: [F; SHA256_WORD_U16S] = array::from_fn(|j| {
                compose::<F>(&small_sig1_field::<F>(&w[i + 2])[j * 16..(j + 1) * 16], 1)
            });
            // w_{t-7}
            let w_7 = if i < 3 {
                local_cols.schedule_helper.w_3[i]
            } else {
                w_limbs[i - 3]
            };
            // w_t
            let w_cur = w_limbs[i + 4];
            for j in 0..SHA256_WORD_U16S {
                let carry = next_cols.message_schedule.carry_or_buffer[i][j * 2]
                    + F::TWO * next_cols.message_schedule.carry_or_buffer[i][j * 2 + 1];
                let sum = sig_w_2[j] + w_7[j] - carry * F::from_canonical_u32(1 << 16) - w_cur[j]
                    + if j > 0 {
                        next_cols.message_schedule.carry_or_buffer[i][j * 2 - 2]
                            + F::from_canonical_u32(2)
                                * next_cols.message_schedule.carry_or_buffer[i][j * 2 - 1]
                    } else {
                        F::ZERO
                    };
                local_cols.schedule_helper.intermed_12[i][j] = -sum;
            }
        }
    }

    /// Puts the correct w_3 in the `next_row`
    pub fn generate_w_3<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        let w = [local_cols.message_schedule.w, next_cols.message_schedule.w].concat();
        let w: Vec<[F; SHA256_WORD_U16S]> = w
            .into_iter()
            .map(|x| array::from_fn(|i| compose::<F>(&x[i * 16..(i + 1) * 16], 1)))
            .collect();
        for i in 0..SHA256_ROUNDS_PER_ROW - 1 {
            for j in 0..SHA256_WORD_U16S {
                next_cols.schedule_helper.w_3[i][j] = w[i + 1][j];
            }
        }
    }

    /// Fills the `next_row` as a padding row
    /// Note: we still need to correctly fill in the hash values, carries and count corrections
    pub fn default_row<F: PrimeField32>(
        self: &Sha256Air,
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        next_cols.flags.is_round_row = F::ZERO;
        next_cols.flags.is_first_4_rows = F::ZERO;
        next_cols.flags.is_digest_row = F::ZERO;

        next_cols.flags.is_last_block = F::ZERO;
        next_cols.flags.global_block_idx = F::ZERO;
        next_cols.flags.row_idx =
            get_flag_pt_array(&self.row_idx_encoder, 17).map(F::from_canonical_u32);
        next_cols.flags.local_block_idx = F::ZERO;

        next_cols.message_schedule.w = [[F::ZERO; SHA256_WORD_BITS]; SHA256_ROUNDS_PER_ROW];
        next_cols.message_schedule.carry_or_buffer =
            [[F::ZERO; SHA256_WORD_U16S * 2]; SHA256_ROUNDS_PER_ROW];

        let hash = SHA256_H
            .map(u32_into_limbs::<SHA256_WORD_BITS>)
            .map(|x| x.map(F::from_canonical_u32));

        for i in 0..SHA256_ROUNDS_PER_ROW {
            next_cols.work_vars.a[i] = hash[SHA256_ROUNDS_PER_ROW - i - 1];
            next_cols.work_vars.e[i] = hash[SHA256_ROUNDS_PER_ROW - i + 3];
        }
        Self::generate_carry_ae(local_cols, next_cols, 17);
    }
}
