use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
    cmp::min,
    marker::PhantomData,
    ops::Range,
};

use openvm_circuit::{
    arch::{
        CustomBorrow, MultiRowLayout, MultiRowMetadata, PreflightExecutor, RecordArena,
        SizedRecord, VmStateMut, *,
    },
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
    utils::next_power_of_two_or_zero, AlignedBytesBorrow,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};
use sha2::{compress256, compress512, digest::generic_array::GenericArray};

use crate::{
    Sha256Config, Sha2BlockHasherConfig, Sha2BlockHasherFiller, Sha2Config, Sha2MainChipConfig,
    Sha2Variant, Sha2VmExecutor, Sha384Config, Sha512Config,
};

impl<F: PrimeField32> TraceFiller<F> for Sha2BlockHasherFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        // TODO
    }
}

// --- old tracegen code ---

/// The trace generation of SHA should be done in two passes.
/// The first pass should do `get_block_trace` for every block and generate the invalid rows through
/// `get_default_row` The second pass should go through all the blocks and call
/// `generate_missing_cells`
impl<C: Sha2BlockHasherConfig> Sha2StepHelper<C> {
    pub fn new() -> Self {
        Self {
            // +1 for dummy (padding) rows
            row_idx_encoder: Encoder::new(C::ROWS_PER_BLOCK + 1, 2, false),
            _phantom: PhantomData,
        }
    }

    /// This function takes the input_message (padding not handled), the previous hash,
    /// and returns the new hash after processing the block input
    pub fn get_block_hash(prev_hash: &[C::Word], input: Vec<u8>) -> Vec<C::Word> {
        debug_assert!(prev_hash.len() == C::HASH_WORDS);
        debug_assert!(input.len() == C::BLOCK_U8S);
        let mut new_hash: [C::Word; 8] = prev_hash.try_into().unwrap();
        match C::VARIANT {
            Sha2Variant::Sha256 => {
                let input_array = [*GenericArray::<u8, sha2::digest::consts::U64>::from_slice(
                    &input,
                )];
                let hash_ptr: &mut [u32; 8] = unsafe { std::mem::transmute(&mut new_hash) };
                compress256(hash_ptr, &input_array);
            }
            Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                let hash_ptr: &mut [u64; 8] = unsafe { std::mem::transmute(&mut new_hash) };
                let input_array = [*GenericArray::<u8, sha2::digest::consts::U128>::from_slice(
                    &input,
                )];
                compress512(hash_ptr, &input_array);
            }
        }
        new_hash.to_vec()
    }

    /// This function takes a C::BLOCK_BITS-bit chunk of the input message (padding not handled),
    /// the previous hash, a flag indicating if it's the last block, the global block index, the
    /// local block index, and the buffer values that will be put in rows 0..4.
    /// Will populate the given `trace` with the trace of the block, where the width of the trace is
    /// `trace_width` and the starting column for the `Sha2Air` is `trace_start_col`.
    /// **Note**: this function only generates some of the required trace. Another pass is required,
    /// refer to [`Self::generate_missing_cells`] for details.
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
    ) {
        #[cfg(debug_assertions)]
        {
            assert!(input.len() == C::BLOCK_WORDS);
            assert!(prev_hash.len() == C::HASH_WORDS);
            assert!(trace.len() == trace_width * C::ROWS_PER_BLOCK);
            assert!(trace_start_col + C::WIDTH <= trace_width);
            if local_block_idx == 0 {
                assert!(*prev_hash == *C::get_h());
            }
        }
        let get_range = |start: usize, len: usize| -> Range<usize> { start..start + len };
        let mut message_schedule = vec![C::Word::from(0); C::ROUNDS_PER_BLOCK];
        message_schedule[..input.len()].copy_from_slice(input);
        let mut work_vars = prev_hash.to_vec();
        for (i, row) in trace.chunks_exact_mut(trace_width).enumerate() {
            // do the rounds
            if i < C::ROUND_ROWS {
                let mut cols: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
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
                if i < C::MESSAGE_ROWS {
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
                        le_limbs_into_word::<C>(
                            cols.message_schedule
                                .w
                                .row(j)
                                .map(|f| f.as_canonical_u32())
                                .as_slice()
                                .unwrap(),
                        ),
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
                let mut cols: Sha2DigestColsRefMut<F> = Sha2DigestColsRefMut::from::<C>(
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
                        get_flag_pt_array(&self.row_idx_encoder, C::ROUND_ROWS)
                            .into_iter()
                            .map(F::from_canonical_u32),
                    )
                    .for_each(|(x, y)| *x = y);

                *cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);

                *cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);
                let final_hash: Vec<C::Word> = (0..C::HASH_WORDS)
                    .map(|i| work_vars[i].wrapping_add(prev_hash[i]))
                    .collect();
                let final_hash_limbs: Vec<Vec<u32>> = final_hash
                    .iter()
                    .map(|word| word_into_u8_limbs::<C>(*word))
                    .collect();
                // need to ensure final hash limbs are bytes, in order for
                //   prev_hash[i] + work_vars[i] == final_hash[i]
                // to be constrained correctly
                for word in final_hash_limbs.iter() {
                    for chunk in word.chunks(2) {
                        bitwise_lookup_chip.request_range(chunk[0], chunk[1]);
                    }
                }
                cols.final_hash
                    .iter_mut()
                    .zip((0..C::HASH_WORDS).flat_map(|i| {
                        word_into_u8_limbs::<C>(final_hash[i])
                            .into_iter()
                            .map(F::from_canonical_u32)
                            .collect::<Vec<_>>()
                    }))
                    .for_each(|(x, y)| *x = y);
                cols.prev_hash
                    .iter_mut()
                    .zip(prev_hash.iter().flat_map(|f| {
                        word_into_u16_limbs::<C>(*f)
                            .into_iter()
                            .map(F::from_canonical_u32)
                            .collect::<Vec<_>>()
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
                            le_limbs_into_word::<C>(
                                f.map(|x| x.as_canonical_u32()).as_slice().unwrap(),
                            )
                        })
                        .map(word_into_bits::<C>)
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
            let mut local_cols: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
                &mut local[get_range(trace_start_col, C::ROUND_WIDTH)],
            );
            let mut next_cols: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
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
                // `next` is a digest row.
                // Fill in `carry_a` and `carry_e` with dummy values so the constraints on `a` and
                // `e` hold.
                let const_local_cols = Sha2RoundColsRef::<F>::from_mut::<C>(&local_cols);
                Self::generate_carry_ae(const_local_cols.clone(), &mut next_cols);
                // Fill in row 16's `intermed_4` with dummy values so the message schedule
                // constraints holds on that row
                Self::generate_intermed_4(const_local_cols, &mut next_cols);
            }
            if i <= 2 {
                // i is in 0..3.
                // Fill in `local.intermed_12` with dummy values so the message schedule constraints
                // hold on rows 1..4.
                Self::generate_intermed_12(
                    &mut local_cols,
                    Sha2RoundColsRef::<F>::from_mut::<C>(&next_cols),
                );
            }
        }
    }

    /// This function will fill in the cells that we couldn't do during the first pass.
    /// This function should be called only after `generate_block_trace` was called for all blocks
    /// And [`Self::generate_default_row`] is called for all invalid rows
    /// Will populate the missing values of `trace`, where the width of the trace is `trace_width`
    /// and the starting column for the `ShaAir` is `trace_start_col`.
    /// Note: `trace` needs to be the rows 1..C::ROWS_PER_BLOCK of a block and the first row of the
    /// next block
    pub fn generate_missing_cells<F: PrimeField32>(
        &self,
        trace: &mut [F],
        trace_width: usize,
        trace_start_col: usize,
    ) {
        let rows = &mut trace[(C::ROUND_ROWS - 2) * trace_width..(C::ROUND_ROWS + 1) * trace_width];
        let (last_round_row, rows) = rows.split_at_mut(trace_width);
        let (digest_row, next_block_first_row) = rows.split_at_mut(trace_width);
        let mut cols_last_round_row: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
            &mut last_round_row[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        let mut cols_digest_row: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
            &mut digest_row[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        let mut cols_next_block_first_row: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
            &mut next_block_first_row[trace_start_col..trace_start_col + C::ROUND_WIDTH],
        );
        // Fill in the last round row's `intermed_12` with dummy values so the message schedule
        // constraints holds on row 16
        Self::generate_intermed_12(
            &mut cols_last_round_row,
            Sha2RoundColsRef::from_mut::<C>(&cols_digest_row),
        );
        // Fill in the digest row's `intermed_12` with dummy values so the message schedule
        // constraints holds on the next block's row 0
        Self::generate_intermed_12(
            &mut cols_digest_row,
            Sha2RoundColsRef::from_mut::<C>(&cols_next_block_first_row),
        );
        // Fill in the next block's first row's `intermed_4` with dummy values so the message
        // schedule constraints holds on that row
        Self::generate_intermed_4(
            Sha2RoundColsRef::from_mut::<C>(&cols_digest_row),
            &mut cols_next_block_first_row,
        );
    }

    /// Fills the `cols` as a padding row
    /// Note: we still need to correctly fill in the hash values, carries and intermeds
    pub fn generate_default_row<F: PrimeField32>(&self, mut cols: Sha2RoundColsRefMut<F>) {
        *cols.flags.is_round_row = F::ZERO;
        *cols.flags.is_first_4_rows = F::ZERO;
        *cols.flags.is_digest_row = F::ZERO;

        *cols.flags.is_last_block = F::ZERO;
        *cols.flags.global_block_idx = F::ZERO;
        cols.flags
            .row_idx
            .iter_mut()
            .zip(
                get_flag_pt_array(&self.row_idx_encoder, C::ROWS_PER_BLOCK)
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

    /// The following functions do the calculations in native field since they will be called on
    /// padding rows which can overflow and we need to make sure it matches the AIR constraints
    /// Puts the correct carries in the `next_row`, the resulting carries can be out of bounds
    pub fn generate_carry_ae<F: PrimeField32>(
        local_cols: Sha2RoundColsRef<F>,
        next_cols: &mut Sha2RoundColsRefMut<F>,
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
            let sig_a = big_sig0_field::<F, C>(a[i + 3].as_slice().unwrap());
            let maj_abc = maj_field::<F>(
                a[i + 3].as_slice().unwrap(),
                a[i + 2].as_slice().unwrap(),
                a[i + 1].as_slice().unwrap(),
            );
            let d = a[i];
            let cur_e = e[i + 4];
            let sig_e = big_sig1_field::<F, C>(e[i + 3].as_slice().unwrap());
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
    fn generate_intermed_4<F: PrimeField32>(
        local_cols: Sha2RoundColsRef<F>,
        next_cols: &mut Sha2RoundColsRefMut<F>,
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
            let sig_w = small_sig0_field::<F, C>(w[i + 1].as_slice().unwrap());
            let sig_w_limbs: Vec<F> = (0..C::WORD_U16S)
                .map(|j| compose::<F>(&sig_w[j * 16..(j + 1) * 16], 1))
                .collect();
            for (j, sig_w_limb) in sig_w_limbs.iter().enumerate() {
                next_cols.schedule_helper.intermed_4[[i, j]] = w_limbs[i][j] + *sig_w_limb;
            }
        }
    }

    /// Puts the needed intermed_12 in the `local_row`
    fn generate_intermed_12<F: PrimeField32>(
        local_cols: &mut Sha2RoundColsRefMut<F>,
        next_cols: Sha2RoundColsRef<F>,
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
                        &small_sig1_field::<F, C>(w[i + 2].as_slice().unwrap())
                            [j * 16..(j + 1) * 16],
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
pub fn generate_trace<F: PrimeField32, C: Sha2BlockHasherConfig>(
    step: &Sha2StepHelper<C>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    width: usize,
    records: Vec<(Vec<u8>, bool)>,
) -> RowMajorMatrix<F> {
    for (input, _) in &records {
        debug_assert!(input.len() == C::BLOCK_U8S);
    }

    let non_padded_height = records.len() * C::ROWS_PER_BLOCK;
    let height = next_power_of_two_or_zero(non_padded_height);
    let mut values = F::zero_vec(height * width);

    struct BlockContext<C: Sha2BlockHasherConfig> {
        prev_hash: Vec<C::Word>, // len is C::HASH_WORDS
        local_block_idx: u32,
        global_block_idx: u32,
        input: Vec<u8>, // len is C::BLOCK_U8S
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
            prev_hash = Sha2StepHelper::<C>::get_block_hash(&prev_hash, input);
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
                    le_limbs_into_word::<C>(
                        &(0..C::WORD_U8S)
                            .map(|j| input[(i + 1) * C::WORD_U8S - j - 1] as u32)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            step.generate_block_trace(
                block,
                width,
                0,
                &input_words,
                bitwise_lookup_chip.clone(),
                &prev_hash,
                is_last_block,
                global_block_idx,
                local_block_idx,
            );
        });
    // second pass: padding rows
    values[width * non_padded_height..]
        .par_chunks_mut(width)
        .for_each(|row| {
            let cols: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(row);
            step.generate_default_row(cols);
        });

    // second pass: non-padding rows
    values[width..]
        .par_chunks_mut(width * C::ROWS_PER_BLOCK)
        .take(non_padded_height / C::ROWS_PER_BLOCK)
        .for_each(|chunk| {
            step.generate_missing_cells(chunk, width, 0);
        });
    RowMajorMatrix::new(values, width)
}
