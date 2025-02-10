use std::{array, borrow::BorrowMut, ops::Range};

use ndarray::{concatenate, ArrayViewMut1};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};
use sha2::{compress256, compress512, digest::generic_array::GenericArray};

use super::{
    air::ShaAir, big_sig0_field, big_sig1_field, ch_field, compose, get_flag_pt_array, maj_field,
    small_sig0_field, small_sig1_field, ShaRoundColsRefMut,
};
use crate::{
    big_sig0, big_sig1, ch, limbs_into_u32, limbs_into_word, maj, small_sig0, small_sig1,
    u32_into_bits, word_into_bits, word_into_u16_limbs, word_into_u8_limbs, ShaConfig,
    ShaDigestColsRefMut, ShaPrecomputedValues, ShaRoundColsRef, WrappingAdd,
};

/// The trace generation of SHA256 should be done in two passes.
/// The first pass should do `get_block_trace` for every block and generate the invalid rows through `get_default_row`
/// The second pass should go through all the blocks and call `generate_missing_values`
impl<C: ShaConfig + ShaPrecomputedValues<C::Word>> ShaAir<C> {
    /// This function takes the input_message (padding not handled), the previous hash,
    /// and returns the new hash after processing the block input
    pub fn get_block_hash(prev_hash: &[C::Word], input: Vec<u8>) -> Vec<C::Word> {
        debug_assert!(prev_hash.len() == C::HASH_WORDS);
        debug_assert!(input.len() == C::BLOCK_U8S);
        let mut new_hash: [C::Word; 8] = prev_hash.try_into().unwrap();
        if C::WORD_BITS == 32 {
            let hash_ptr: &mut [u32; 8] = unsafe { std::mem::transmute(&mut new_hash) };
            let input_array = [*GenericArray::<u8, sha2::digest::consts::U64>::from_slice(
                &input,
            )];
            compress256(hash_ptr, &input_array);
        } else if C::WORD_BITS == 64 {
            let hash_ptr: &mut [u64; 8] = unsafe { std::mem::transmute(&mut new_hash) };
            let input_array = [*GenericArray::<u8, sha2::digest::consts::U128>::from_slice(
                &input,
            )];
            compress512(hash_ptr, &input_array);
        }
        new_hash.to_vec()
    }

    /// This function takes a 512-bit chunk of the input message (padding not handled), the previous hash,
    /// a flag indicating if it's the last block, the global block index, the local block index,
    /// and the buffer values that will be put in rows 0..4.
    /// Will populate the given `trace` with the trace of the block, where the width of the trace is `trace_width`
    /// and the starting column for the `Sha256Air` is `trace_start_col`.
    /// **Note**: this function only generates some of the required trace. Another pass is required, refer to [`Self::generate_missing_cells`] for details.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_block_trace<F: PrimeField32>(
        &self,
        trace: &mut [F],
        trace_width: usize,
        trace_start_col: usize,
        input: &[C::Word],
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        prev_hash: &[C::Word],
        is_last_block: bool,
        global_block_idx: u32,
        local_block_idx: u32,
        buffer_vals: &[&[F]; 4],
    ) {
        debug_assert!(input.len() == C::BLOCK_WORDS);
        debug_assert!(prev_hash.len() == C::HASH_WORDS);
        debug_assert!(buffer_vals.iter().all(|x| x.len() == C::BUFFER_SIZE));
        #[cfg(debug_assertions)]
        {
            assert!(trace.len() == trace_width * C::ROWS_PER_BLOCK);
            assert!(trace_start_col + C::WIDTH <= trace_width);
            assert!(self.bitwise_lookup_bus == bitwise_lookup_chip.bus());
            if local_block_idx == 0 {
                assert!(*prev_hash == *C::get_h());
            }
        }
        let get_range = |start: usize, len: usize| -> Range<usize> { start..start + len };
        let mut message_schedule = vec![C::Word::from(0); C::ROUNDS_PER_BLOCK];
        message_schedule[..input.len()].copy_from_slice(input);
        let mut work_vars = prev_hash.to_vec();
        for (i, row) in trace.chunks_exact_mut(trace_width).enumerate() {
            // TODO: sha512
            // doing the 64 rounds in 16 rows
            if i < 16 {
                let mut cols: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
                    &mut row[get_range(trace_start_col, C::ROUND_WIDTH)],
                );
                *cols.flags.is_round_row = F::ONE;
                *cols.flags.is_first_4_rows = if i < 4 { F::ONE } else { F::ZERO };
                *cols.flags.is_digest_row = F::ZERO;
                *cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags
                    .row_idx
                    .iter_mut()
                    .zip(
                        get_flag_pt_array(&self.row_idx_encoder, i)
                            .into_iter()
                            .map(F::from_canonical_u32),
                    )
                    .for_each(|(x, y)| *x = y);

                *cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);
                *cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);

                // W_idx = M_idx
                // TODO: fix this. should be smtg like `if i < C::BLOCK_WORDS`
                if i < C::ROWS_PER_BLOCK / C::ROUNDS_PER_ROW {
                    for j in 0..C::ROUNDS_PER_ROW {
                        cols.message_schedule
                            .w
                            .row_mut(j)
                            .iter_mut()
                            .zip(
                                word_into_bits::<C>(input[i * C::ROUNDS_PER_ROW + j])
                                    .into_iter()
                                    .map(F::from_canonical_u32),
                            )
                            .for_each(|(x, y)| *x = y);
                        cols.message_schedule
                            .carry_or_buffer
                            .row_mut(j)
                            .iter_mut()
                            .zip(
                                (0..C::WORD_U16S).map(|k| buffer_vals[i][j * C::WORD_U16S * 2 + k]),
                            )
                            .for_each(|(x, y)| *x = y);
                    }
                }
                // W_idx = SIG1(W_{idx-2}) + W_{idx-7} + SIG0(W_{idx-15}) + W_{idx-16}
                else {
                    for j in 0..C::ROUNDS_PER_ROW {
                        let idx = i * C::ROUNDS_PER_ROW + j;
                        let nums: [C::Word; 4] = [
                            small_sig1::<C>(message_schedule[idx - 2]),
                            message_schedule[idx - 7],
                            small_sig0::<C>(message_schedule[idx - 15]),
                            message_schedule[idx - 16],
                        ];
                        let w: C::Word = nums
                            .iter()
                            .fold(C::Word::from(0), |acc, &num| acc.wrapping_add(num));
                        cols.message_schedule
                            .w
                            .row_mut(j)
                            .iter_mut()
                            .zip(
                                word_into_bits::<C>(w)
                                    .into_iter()
                                    .map(F::from_canonical_u32),
                            )
                            .for_each(|(x, y)| *x = y);

                        let nums_limbs = nums
                            .iter()
                            .map(|x| word_into_u16_limbs::<C>(*x))
                            .collect::<Vec<_>>();
                        let w_limbs = word_into_u16_limbs::<C>(w);

                        // fill in the carrys
                        for k in 0..C::WORD_U16S {
                            let mut sum = nums_limbs.iter().fold(0, |acc, num| acc + num[k]);
                            if k > 0 {
                                sum += (cols.message_schedule.carry_or_buffer[[j, k * 2 - 2]]
                                    + F::TWO
                                        * cols.message_schedule.carry_or_buffer[[j, k * 2 - 1]])
                                .as_canonical_u32();
                            }
                            let carry = (sum - w_limbs[k]) >> 16;
                            cols.message_schedule.carry_or_buffer[[j, k * 2]] =
                                F::from_canonical_u32(carry & 1);
                            cols.message_schedule.carry_or_buffer[[j, k * 2 + 1]] =
                                F::from_canonical_u32(carry >> 1);
                        }
                        // update the message schedule
                        message_schedule[idx] = w;
                    }
                }
                // fill in the work variables
                for j in 0..C::ROUNDS_PER_ROW {
                    // t1 = h + SIG1(e) + ch(e, f, g) + K_idx + W_idx
                    let t1 = [
                        work_vars[7],
                        big_sig1::<C>(work_vars[4]),
                        ch::<C>(work_vars[4], work_vars[5], work_vars[6]),
                        C::get_k()[i * C::ROUNDS_PER_ROW + j],
                        limbs_into_u32(
                            cols.message_schedule
                                .w
                                .row(j)
                                .map(|f| f.as_canonical_u32())
                                .as_slice()
                                .unwrap(),
                        )
                        .into(),
                    ];
                    let t1_sum: C::Word = t1
                        .iter()
                        .fold(C::Word::from(0), |acc, &num| acc.wrapping_add(num));

                    // t2 = SIG0(a) + maj(a, b, c)
                    let t2 = [
                        big_sig0::<C>(work_vars[0]),
                        maj::<C>(work_vars[0], work_vars[1], work_vars[2]),
                    ];

                    let t2_sum: C::Word = t2
                        .iter()
                        .fold(C::Word::from(0), |acc, &num| acc.wrapping_add(num));

                    // e = d + t1
                    let e = work_vars[3].wrapping_add(t1_sum);
                    cols.work_vars
                        .e
                        .row_mut(j)
                        .iter_mut()
                        .zip(
                            word_into_bits::<C>(e)
                                .into_iter()
                                .map(F::from_canonical_u32),
                        )
                        .for_each(|(x, y)| *x = y);
                    let e_limbs = word_into_u16_limbs::<C>(e);
                    // a = t1 + t2
                    let a = t1_sum.wrapping_add(t2_sum);
                    cols.work_vars
                        .a
                        .row_mut(j)
                        .iter_mut()
                        .zip(
                            word_into_bits::<C>(a)
                                .into_iter()
                                .map(F::from_canonical_u32),
                        )
                        .for_each(|(x, y)| *x = y);
                    let a_limbs = word_into_u16_limbs::<C>(a);
                    // fill in the carrys
                    for k in 0..C::WORD_U16S {
                        let t1_limb = t1
                            .iter()
                            .fold(0, |acc, &num| acc + word_into_u16_limbs::<C>(num)[k]);
                        let t2_limb = t2
                            .iter()
                            .fold(0, |acc, &num| acc + word_into_u16_limbs::<C>(num)[k]);

                        let mut e_limb = t1_limb + word_into_u16_limbs::<C>(work_vars[3])[k];
                        let mut a_limb = t1_limb + t2_limb;
                        if k > 0 {
                            a_limb += cols.work_vars.carry_a[[j, k - 1]].as_canonical_u32();
                            e_limb += cols.work_vars.carry_e[[j, k - 1]].as_canonical_u32();
                        }
                        let carry_a = (a_limb - a_limbs[k]) >> 16;
                        let carry_e = (e_limb - e_limbs[k]) >> 16;
                        cols.work_vars.carry_a[[j, k]] = F::from_canonical_u32(carry_a);
                        cols.work_vars.carry_e[[j, k]] = F::from_canonical_u32(carry_e);
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

                // filling w_3 and intermed_4 here and the rest later
                if i > 0 {
                    for j in 0..C::ROUNDS_PER_ROW {
                        let idx = i * C::ROUNDS_PER_ROW + j;
                        let w_4 = word_into_u16_limbs::<C>(message_schedule[idx - 4]);
                        let sig_0_w_3 =
                            word_into_u16_limbs::<C>(small_sig0::<C>(message_schedule[idx - 3]));
                        cols.schedule_helper
                            .intermed_4
                            .row_mut(j)
                            .iter_mut()
                            .zip(
                                (0..C::WORD_U16S)
                                    .map(|k| F::from_canonical_u32(w_4[k] + sig_0_w_3[k]))
                                    .collect::<Vec<_>>(),
                            )
                            .for_each(|(x, y)| *x = y);
                        if j < C::ROUNDS_PER_ROW - 1 {
                            let w_3 = message_schedule[idx - 3];
                            cols.schedule_helper
                                .w_3
                                .row_mut(j)
                                .iter_mut()
                                .zip(
                                    word_into_u16_limbs::<C>(w_3)
                                        .into_iter()
                                        .map(F::from_canonical_u32),
                                )
                                .for_each(|(x, y)| *x = y);
                        }
                    }
                }
            }
            // generate the digest row
            else {
                let mut cols: ShaDigestColsRefMut<F> = ShaDigestColsRefMut::from::<C>(
                    &mut row[get_range(trace_start_col, C::DIGEST_WIDTH)],
                );
                for j in 0..C::ROUNDS_PER_ROW - 1 {
                    let w_3 = message_schedule[i * C::ROUNDS_PER_ROW + j - 3];
                    cols.schedule_helper
                        .w_3
                        .row_mut(j)
                        .iter_mut()
                        .zip(
                            word_into_u16_limbs::<C>(w_3)
                                .into_iter()
                                .map(F::from_canonical_u32)
                                .collect::<Vec<_>>(),
                        )
                        .for_each(|(x, y)| *x = y);
                }
                *cols.flags.is_round_row = F::ZERO;
                *cols.flags.is_first_4_rows = F::ZERO;
                *cols.flags.is_digest_row = F::ONE;
                *cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags
                    .row_idx
                    .iter_mut()
                    .zip(
                        get_flag_pt_array(&self.row_idx_encoder, 16)
                            .into_iter()
                            .map(F::from_canonical_u32),
                    )
                    .for_each(|(x, y)| *x = y);

                *cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);

                *cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);
                let final_hash: Vec<C::Word> = (0..C::HASH_WORDS)
                    .map(|i| work_vars[i].wrapping_add(prev_hash[i]))
                    .collect();
                cols.final_hash
                    .iter_mut()
                    .zip((0..C::HASH_WORDS).flat_map(|i| {
                        word_into_u8_limbs::<C>(final_hash[i])
                            .into_iter()
                            .map(F::from_canonical_u32)
                    }))
                    .for_each(|(x, y)| *x = y);
                cols.prev_hash
                    .iter_mut()
                    .zip(prev_hash.into_iter().flat_map(|f| {
                        word_into_u16_limbs::<C>(*f)
                            .into_iter()
                            .map(F::from_canonical_u32)
                    }))
                    .for_each(|(x, y)| *x = y);
                let hash = if is_last_block {
                    C::get_h()
                        .iter()
                        .map(|x| word_into_bits::<C>(*x))
                        .collect::<Vec<_>>()
                } else {
                    cols.final_hash
                        .rows_mut()
                        .into_iter()
                        .map(|f| {
                            limbs_into_u32(f.map(|x| x.as_canonical_u32()).as_slice().unwrap())
                        })
                        .map(u32_into_bits::<C>)
                        .collect()
                }
                .into_iter()
                .map(|x| x.into_iter().map(F::from_canonical_u32))
                .collect::<Vec<_>>();

                for i in 0..C::ROUNDS_PER_ROW {
                    cols.hash
                        .a
                        .row_mut(i)
                        .iter_mut()
                        .zip(hash[C::ROUNDS_PER_ROW - i - 1].clone())
                        .for_each(|(x, y)| *x = y);
                    cols.hash
                        .e
                        .row_mut(i)
                        .iter_mut()
                        .zip(hash[C::ROUNDS_PER_ROW - i + 3].clone())
                        .for_each(|(x, y)| *x = y);
                }
            }
        }

        for i in 0..C::ROWS_PER_BLOCK - 1 {
            let rows = &mut trace[i * trace_width..(i + 2) * trace_width];
            let (local, next) = rows.split_at_mut(trace_width);
            let mut local_cols: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
                &mut local[get_range(trace_start_col, C::ROUND_WIDTH)],
            );
            let mut next_cols: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
                &mut next[get_range(trace_start_col, C::ROUND_WIDTH)],
            );
            if i > 0 {
                for j in 0..C::ROUNDS_PER_ROW {
                    next_cols
                        .schedule_helper
                        .intermed_8
                        .row_mut(j)
                        .assign(&local_cols.schedule_helper.intermed_4.row(j));
                    if (2..C::ROWS_PER_BLOCK - 3).contains(&i) {
                        next_cols
                            .schedule_helper
                            .intermed_12
                            .row_mut(j)
                            .assign(&local_cols.schedule_helper.intermed_8.row(j));
                    }
                }
            }
            if i == C::ROWS_PER_BLOCK - 2 {
                let const_ref = ShaRoundColsRef::<F>::from_mut::<C>(&local_cols);
                Self::generate_carry_ae(const_ref.clone(), &mut next_cols);
                Self::generate_intermed_4(const_ref, &mut next_cols);
            }
            if i <= 2 {
                Self::generate_intermed_12(
                    &mut local_cols,
                    ShaRoundColsRef::<F>::from_mut::<C>(&next_cols),
                );
            }
        }
    }

    /// This function will fill in the cells that we couldn't do during the first pass.
    /// This function should be called only after `generate_block_trace` was called for all blocks
    /// And [`Self::generate_default_row`] is called for all invalid rows
    /// Will populate the missing values of `trace`, where the width of the trace is `trace_width`
    /// and the starting column for the `Sha256Air` is `trace_start_col`.
    /// Note: `trace` needs to be the rows 1..17 of a block and the first row of the next block
    pub fn generate_missing_cells<F: PrimeField32>(
        &self,
        trace: &mut [F],
        trace_width: usize,
        trace_start_col: usize,
    ) {
        // Here row_17 = next blocks row 0
        let rows_15_17 = &mut trace[14 * trace_width..17 * trace_width];
        let (row_15, row_16_17) = rows_15_17.split_at_mut(trace_width);
        let (row_16, row_17) = row_16_17.split_at_mut(trace_width);
        let mut cols_15: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
            &mut row_15[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        let mut cols_16: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
            &mut row_16[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        let mut cols_17: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(
            &mut row_17[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        Self::generate_intermed_12(&mut cols_15, ShaRoundColsRef::from_mut::<C>(&cols_16));
        Self::generate_intermed_12(&mut cols_16, ShaRoundColsRef::from_mut::<C>(&cols_17));
        Self::generate_intermed_4(ShaRoundColsRef::from_mut::<C>(&cols_16), &mut cols_17);
    }

    /// Fills the `cols` as a padding row
    /// Note: we still need to correctly fill in the hash values, carries and intermeds
    pub fn generate_default_row<F: PrimeField32>(&self, mut cols: ShaRoundColsRefMut<F>) {
        *cols.flags.is_round_row = F::ZERO;
        *cols.flags.is_first_4_rows = F::ZERO;
        *cols.flags.is_digest_row = F::ZERO;

        *cols.flags.is_last_block = F::ZERO;
        *cols.flags.global_block_idx = F::ZERO;
        cols.flags
            .row_idx
            .iter_mut()
            .zip(
                get_flag_pt_array(&self.row_idx_encoder, 17)
                    .into_iter()
                    .map(F::from_canonical_u32),
            )
            .for_each(|(x, y)| *x = y);
        *cols.flags.local_block_idx = F::ZERO;

        cols.message_schedule
            .w
            .iter_mut()
            .for_each(|x| *x = F::ZERO);
        cols.message_schedule
            .carry_or_buffer
            .iter_mut()
            .for_each(|x| *x = F::ZERO);

        let hash = C::get_h()
            .iter()
            .map(|x| word_into_bits::<C>(*x))
            .map(|x| x.into_iter().map(F::from_canonical_u32).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for i in 0..C::ROUNDS_PER_ROW {
            cols.work_vars
                .a
                .row_mut(i)
                .iter_mut()
                .zip(hash[C::ROUNDS_PER_ROW - i - 1].clone())
                .for_each(|(x, y)| *x = y);
            cols.work_vars
                .e
                .row_mut(i)
                .iter_mut()
                .zip(hash[C::ROUNDS_PER_ROW - i + 3].clone())
                .for_each(|(x, y)| *x = y);
        }

        cols.work_vars
            .carry_a
            .iter_mut()
            .zip((0..C::ROUNDS_PER_ROW).flat_map(|i| {
                (0..C::WORD_U16S)
                    .map(|j| F::from_canonical_u32(C::get_invalid_carry_a(i)[j]))
                    .collect::<Vec<_>>()
            }))
            .for_each(|(x, y)| *x = y);
        cols.work_vars
            .carry_e
            .iter_mut()
            .zip((0..C::ROUNDS_PER_ROW).flat_map(|i| {
                (0..C::WORD_U16S)
                    .map(|j| F::from_canonical_u32(C::get_invalid_carry_e(i)[j]))
                    .collect::<Vec<_>>()
            }))
            .for_each(|(x, y)| *x = y);
    }

    /// The following functions do the calculations in native field since they will be called on padding rows
    /// which can overflow and we need to make sure it matches the AIR constraints
    /// Puts the correct carrys in the `next_row`, the resulting carrys can be out of bound
    fn generate_carry_ae<'a, 'b, F: PrimeField32>(
        local_cols: ShaRoundColsRef<'a, F>,
        next_cols: &mut ShaRoundColsRefMut<'b, F>,
    ) {
        let a = [
            local_cols
                .work_vars
                .a
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
            next_cols.work_vars.a.rows().into_iter().collect::<Vec<_>>(),
        ]
        .concat();
        let e = [
            local_cols
                .work_vars
                .e
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
            next_cols.work_vars.e.rows().into_iter().collect::<Vec<_>>(),
        ]
        .concat();
        for i in 0..C::ROUNDS_PER_ROW {
            let cur_a = a[i + 4];
            let sig_a = big_sig0_field::<F>(a[i + 3].as_slice().unwrap());
            let maj_abc = maj_field::<F>(
                a[i + 3].as_slice().unwrap(),
                a[i + 2].as_slice().unwrap(),
                a[i + 1].as_slice().unwrap(),
            );
            let d = a[i];
            let cur_e = e[i + 4];
            let sig_e = big_sig1_field::<F>(e[i + 3].as_slice().unwrap());
            let ch_efg = ch_field::<F>(
                e[i + 3].as_slice().unwrap(),
                e[i + 2].as_slice().unwrap(),
                e[i + 1].as_slice().unwrap(),
            );
            let h = e[i];

            let t1 = [h.to_vec(), sig_e, ch_efg.to_vec()];
            let t2 = [sig_a, maj_abc];
            for j in 0..C::WORD_U16S {
                let t1_limb_sum = t1.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let t2_limb_sum = t2.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let d_limb = compose::<F>(&d.as_slice().unwrap()[j * 16..(j + 1) * 16], 1);
                let cur_a_limb = compose::<F>(&cur_a.as_slice().unwrap()[j * 16..(j + 1) * 16], 1);
                let cur_e_limb = compose::<F>(&cur_e.as_slice().unwrap()[j * 16..(j + 1) * 16], 1);
                let sum = d_limb
                    + t1_limb_sum
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_e[[i, j - 1]]
                    }
                    - cur_e_limb;
                let carry_e = sum * (F::from_canonical_u32(1 << 16).inverse());

                let sum = t1_limb_sum
                    + t2_limb_sum
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_a[[i, j - 1]]
                    }
                    - cur_a_limb;
                let carry_a = sum * (F::from_canonical_u32(1 << 16).inverse());
                next_cols.work_vars.carry_e[[i, j]] = carry_e;
                next_cols.work_vars.carry_a[[i, j]] = carry_a;
            }
        }
    }

    /// Puts the correct intermed_4 in the `next_row`
    fn generate_intermed_4<'a, 'b, F: PrimeField32>(
        local_cols: ShaRoundColsRef<'a, F>,
        next_cols: &mut ShaRoundColsRefMut<'b, F>,
    ) {
        let w = [
            local_cols
                .message_schedule
                .w
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
            next_cols
                .message_schedule
                .w
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
        ]
        .concat();
        let w_limbs: Vec<Vec<F>> = w
            .iter()
            .map(|x| {
                (0..C::WORD_U16S)
                    .map(|i| compose::<F>(&x.as_slice().unwrap()[i * 16..(i + 1) * 16], 1))
                    .collect::<Vec<F>>()
            })
            .collect();
        for i in 0..C::ROUNDS_PER_ROW {
            let sig_w = small_sig0_field::<F>(w[i + 1].as_slice().unwrap());
            let sig_w_limbs: Vec<F> = (0..C::WORD_U16S)
                .map(|j| compose::<F>(&sig_w[j * 16..(j + 1) * 16], 1))
                .collect();
            for (j, sig_w_limb) in sig_w_limbs.iter().enumerate() {
                next_cols.schedule_helper.intermed_4[[i, j]] = w_limbs[i][j] + *sig_w_limb;
            }
        }
    }

    /// Puts the needed intermed_12 in the `local_row`
    fn generate_intermed_12<'a, 'b, F: PrimeField32>(
        local_cols: &mut ShaRoundColsRefMut<'a, F>,
        next_cols: ShaRoundColsRef<'b, F>,
    ) {
        let w = [
            local_cols
                .message_schedule
                .w
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
            next_cols
                .message_schedule
                .w
                .rows()
                .into_iter()
                .collect::<Vec<_>>(),
        ]
        .concat();
        let w_limbs: Vec<Vec<F>> = w
            .iter()
            .map(|x| {
                (0..C::WORD_U16S)
                    .map(|i| compose::<F>(&x.as_slice().unwrap()[i * 16..(i + 1) * 16], 1))
                    .collect::<Vec<F>>()
            })
            .collect();
        for i in 0..C::ROUNDS_PER_ROW {
            // sig_1(w_{t-2})
            let sig_w_2: Vec<F> = (0..C::WORD_U16S)
                .map(|j| {
                    compose::<F>(
                        &small_sig1_field::<F>(w[i + 2].as_slice().unwrap())[j * 16..(j + 1) * 16],
                        1,
                    )
                })
                .collect();
            // w_{t-7}
            let w_7 = if i < 3 {
                local_cols.schedule_helper.w_3.row(i).to_slice().unwrap()
            } else {
                w_limbs[i - 3].as_slice()
            };
            // w_t
            let w_cur = w_limbs[i + 4].as_slice();
            for j in 0..C::WORD_U16S {
                let carry = next_cols.message_schedule.carry_or_buffer[[i, j * 2]]
                    + F::TWO * next_cols.message_schedule.carry_or_buffer[[i, j * 2 + 1]];
                let sum = sig_w_2[j] + w_7[j] - carry * F::from_canonical_u32(1 << 16) - w_cur[j]
                    + if j > 0 {
                        next_cols.message_schedule.carry_or_buffer[[i, j * 2 - 2]]
                            + F::from_canonical_u32(2)
                                * next_cols.message_schedule.carry_or_buffer[[i, j * 2 - 1]]
                    } else {
                        F::ZERO
                    };
                local_cols.schedule_helper.intermed_12[[i, j]] = -sum;
            }
        }
    }
}

/// `records` consists of pairs of `(input_block, is_last_block)`.
pub fn generate_trace<F: PrimeField32, C: ShaConfig + ShaPrecomputedValues<C::Word>>(
    sub_air: &ShaAir<C>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    records: Vec<(Vec<u8>, bool)>,
) -> RowMajorMatrix<F> {
    for (input, _) in &records {
        debug_assert!(input.len() == C::BLOCK_U8S);
    }

    let non_padded_height = records.len() * C::ROWS_PER_BLOCK;
    let height = next_power_of_two_or_zero(non_padded_height);
    let width = <ShaAir<C> as BaseAir<F>>::width(sub_air);
    let mut values = F::zero_vec(height * width);

    struct BlockContext<C: ShaConfig> {
        prev_hash: Vec<C::Word>, // HASH_WORDS
        local_block_idx: u32,
        global_block_idx: u32,
        input: Vec<u8>, // BLOCK_U8S
        is_last_block: bool,
    }
    let mut block_ctx: Vec<BlockContext<C>> = Vec::with_capacity(records.len());
    let mut prev_hash = C::get_h().to_vec();
    let mut local_block_idx = 0;
    let mut global_block_idx = 1;
    for (input, is_last_block) in records {
        block_ctx.push(BlockContext {
            prev_hash: prev_hash.clone(),
            local_block_idx,
            global_block_idx,
            input: input.clone(),
            is_last_block,
        });
        global_block_idx += 1;
        if is_last_block {
            local_block_idx = 0;
            prev_hash = C::get_h().to_vec();
        } else {
            local_block_idx += 1;
            prev_hash = ShaAir::<C>::get_block_hash(&prev_hash, input);
        }
    }
    // first pass
    values
        .par_chunks_exact_mut(width * C::ROWS_PER_BLOCK)
        .zip(block_ctx)
        .for_each(|(block, ctx)| {
            let BlockContext {
                prev_hash,
                local_block_idx,
                global_block_idx,
                input,
                is_last_block,
            } = ctx;
            let input_words = (0..C::BLOCK_WORDS)
                .map(|i| {
                    limbs_into_word::<C>(
                        &input[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                            .iter()
                            .map(|x| *x as u32)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let empty_buffer = vec![F::ZERO; C::BUFFER_SIZE];
            let buffer_vals = [empty_buffer.as_slice(); 4];
            sub_air.generate_block_trace(
                block,
                width,
                0,
                &input_words,
                bitwise_lookup_chip.clone(),
                &prev_hash,
                is_last_block,
                global_block_idx,
                local_block_idx,
                &buffer_vals,
            );
        });
    // second pass: padding rows
    values[width * non_padded_height..]
        .par_chunks_mut(width)
        .for_each(|row| {
            let cols: ShaRoundColsRefMut<F> = ShaRoundColsRefMut::from::<C>(row);
            sub_air.generate_default_row(cols);
        });
    // second pass: non-padding rows
    values[width..]
        .par_chunks_mut(width * C::ROWS_PER_BLOCK)
        .take(non_padded_height / C::ROWS_PER_BLOCK)
        .for_each(|chunk| {
            sub_air.generate_missing_cells(chunk, width, 0);
        });
    RowMajorMatrix::new(values, width)
}
