use std::slice;

use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::LocalOpcode;
use openvm_sha2_air::{Sha2BlockHasherFillerHelper, Sha2RoundColsRef, Sha2RoundColsRefMut};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use crate::{
    Sha2BlockHasherChip, Sha2BlockHasherRoundColsRefMut, Sha2BlockHasherVmConfig, Sha2Config,
    INNER_OFFSET,
};

struct Sha2BlockTraceInput<'a> {
    message_bytes: &'a [u8],
    prev_state: &'a [u8],
}

pub(crate) fn generate_trace_from_postflight<F, C>(
    chip: &Sha2BlockHasherChip<F, C>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError>
where
    F: PrimeField32,
    C: Sha2Config,
{
    let steps = postflight.steps(C::OPCODE.global_opcode());
    let replay_rows = steps
        .par_iter()
        .map(|&step| {
            crate::replay_sha2_from_postflight::<F, C>(postflight, step, chip.pointer_max_bits)
        })
        .collect::<Result<Vec<_>, _>>()?;
    chip.generate_trace_from_replays(&replay_rows)
}

#[cfg(test)]
pub(crate) fn generate_trace_from_postflights<F, C>(
    chip: &Sha2BlockHasherChip<F, C>,
    postflights: &[Postflight<'_, F>],
) -> Result<RowMajorMatrix<F>, PostflightError>
where
    F: PrimeField32,
    C: Sha2Config,
{
    let mut replay_rows = Vec::new();
    for postflight in postflights {
        for &step in postflight.steps(C::OPCODE.global_opcode()) {
            replay_rows.push(crate::replay_sha2_from_postflight::<F, C>(
                postflight,
                step,
                chip.pointer_max_bits,
            )?);
        }
    }
    chip.generate_trace_from_replays(&replay_rows)
}

impl<F, C> Sha2BlockHasherChip<F, C>
where
    F: PrimeField32,
    C: Sha2BlockHasherVmConfig,
{
    fn generate_trace_from_replays(
        &self,
        replay_rows: &[crate::Sha2ReplayRow],
    ) -> Result<RowMajorMatrix<F>, PostflightError> {
        let rows_used = replay_rows
            .len()
            .checked_mul(C::ROWS_PER_BLOCK)
            .ok_or_else(|| PostflightError::new("SHA-2 block-hasher trace height overflow"))?;
        let height = next_power_of_two_or_zero(rows_used);
        let mut trace = RowMajorMatrix::new(
            F::zero_vec(height * C::BLOCK_HASHER_WIDTH),
            C::BLOCK_HASHER_WIDTH,
        );
        let inputs = replay_rows
            .iter()
            .map(|replay| Sha2BlockTraceInput {
                message_bytes: &replay.message_bytes,
                prev_state: &replay.prev_state,
            })
            .collect::<Vec<_>>();
        self.fill_trace_from_inputs(&mut trace, &inputs);
        Ok(trace)
    }

    fn fill_trace_from_inputs(
        &self,
        trace_matrix: &mut RowMajorMatrix<F>,
        inputs: &[Sha2BlockTraceInput<'_>],
    ) {
        if inputs.is_empty() {
            return;
        }

        let rows_used = inputs.len() * C::ROWS_PER_BLOCK;
        let trace = &mut trace_matrix.values[..];
        let prev_hashes = inputs
            .par_iter()
            .map(|input| {
                (0..C::HASH_WORDS)
                    .map(|i| {
                        input.prev_state[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                            .iter()
                            .rev()
                            .fold(C::Word::from(0), |word, &byte| {
                                (word << 8) | u32::from(byte).into()
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // zip the prev_hashes with the next block's prev_hash
        let prev_hashes_and_next_block_prev_hashes = prev_hashes.par_iter().zip(
            prev_hashes[1..]
                .par_iter()
                .chain(prev_hashes[..1].par_iter()),
        );

        // fill in used rows
        trace[..rows_used * C::BLOCK_HASHER_WIDTH]
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH * C::ROWS_PER_BLOCK)
            .zip(
                inputs
                    .par_iter()
                    .zip(prev_hashes_and_next_block_prev_hashes),
            )
            .enumerate()
            .for_each(
                |(block_idx, (block_slice, (input, (prev_hash, next_block_prev_hash))))| {
                    self.fill_block_trace(
                        block_slice,
                        input.message_bytes,
                        block_idx + 1, // 1-indexed
                        prev_hash,
                        next_block_prev_hash,
                        block_idx,
                    );
                },
            );

        // fill in the first dummy row.
        // we need to do this first, so we can compute the carries that make the
        // constraint_word_addition constraints hold on dummy rows (or more precisely, on rows such
        // that the next row is a dummy row).
        let num_blocks = rows_used / C::ROWS_PER_BLOCK;
        let first_dummy_row_cols_const = self.fill_first_dummy_row(
            &mut trace[rows_used * C::BLOCK_HASHER_WIDTH..(rows_used + 1) * C::BLOCK_HASHER_WIDTH],
            &prev_hashes[0],
            num_blocks,
        );

        // fill in the rest of the dummy rows
        let padding_global_block_idx = (num_blocks + 1) as u32;
        trace[(rows_used + 1) * C::BLOCK_HASHER_WIDTH..]
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
            .for_each(|row| {
                // copy the carries from the first dummy row into the current dummy row
                self.inner.generate_default_row(
                    &mut Sha2RoundColsRefMut::from::<C>(
                        &mut row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
                    ),
                    &prev_hashes[0],
                    Some(
                        first_dummy_row_cols_const
                            .work_vars
                            .carry_a
                            .as_slice()
                            .unwrap(),
                    ),
                    Some(
                        first_dummy_row_cols_const
                            .work_vars
                            .carry_e
                            .as_slice()
                            .unwrap(),
                    ),
                    padding_global_block_idx,
                );
            });

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        trace[C::BLOCK_HASHER_WIDTH..]
            .par_chunks_mut(C::BLOCK_HASHER_WIDTH * C::ROWS_PER_BLOCK)
            .take(rows_used / C::ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.inner
                    .generate_missing_cells(chunk, C::BLOCK_HASHER_WIDTH, INNER_OFFSET);
            });

        self.fill_wraparound(trace);
    }

    /// Fill in dummy values for the wrap-around (last row → first row) so that
    /// unconditional constraints hold:
    /// - `intermed_4` on row 0 for the message schedule sigma constraint
    /// - `intermed_12` on the last row for the message schedule addition constraint
    fn fill_wraparound(&self, trace: &mut [F]) {
        let height = trace.len() / C::BLOCK_HASHER_WIDTH;
        let last_row_start = (height - 1) * C::BLOCK_HASHER_WIDTH;

        // Fill intermed_4 on the first row (needs first_row mut, last_row immut)
        {
            let (first_row, rest) = trace.split_at_mut(C::BLOCK_HASHER_WIDTH);
            let last_row = &rest[(height - 2) * C::BLOCK_HASHER_WIDTH..];
            let local = Sha2RoundColsRef::from::<C>(
                &last_row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            let mut next = Sha2RoundColsRefMut::from::<C>(
                &mut first_row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            Sha2BlockHasherFillerHelper::<C>::generate_intermed_4(local, &mut next);
        }

        // Fill intermed_12 on the last row (needs last_row mut, first_row immut)
        {
            let (first_row, rest) = trace.split_at_mut(C::BLOCK_HASHER_WIDTH);
            let next = Sha2RoundColsRef::from::<C>(
                &first_row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            let last_row = &mut rest[last_row_start - C::BLOCK_HASHER_WIDTH..last_row_start];
            let mut local = Sha2RoundColsRefMut::from::<C>(
                &mut last_row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            Sha2BlockHasherFillerHelper::<C>::generate_intermed_12(&mut local, next);
        }
    }

    fn fill_first_dummy_row(
        &self,
        first_dummy_row_mut: &mut [F],
        first_block_prev_hash: &[C::Word],
        num_blocks: usize,
    ) -> Sha2RoundColsRef<'_, F> {
        let first_dummy_row_const =
            unsafe { slice::from_raw_parts(first_dummy_row_mut.as_ptr(), C::BLOCK_HASHER_WIDTH) };
        let first_dummy_row_cols_const = Sha2RoundColsRef::from::<C>(
            &first_dummy_row_const[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
        );

        let first_dummy_row_mut = unsafe {
            slice::from_raw_parts_mut(first_dummy_row_mut.as_mut_ptr(), C::BLOCK_HASHER_WIDTH)
        };
        let mut first_dummy_row_cols_mut: Sha2RoundColsRefMut<F> = Sha2RoundColsRefMut::from::<C>(
            &mut first_dummy_row_mut[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
        );

        // first, fill in everything but the carries into the first dummy row (i.e. fill in the
        // work vars, row_idx, and global_block_idx)
        self.inner.generate_default_row(
            &mut first_dummy_row_cols_mut,
            first_block_prev_hash,
            None,
            None,
            (num_blocks + 1) as u32,
        );

        // Now, this will fill in the first dummy row with the correct carries.
        // This works because we already filled in the work vars into the first dummy row, and
        // generate_carry_ae only looks at the work vars.
        // Note that these carries will work for any pair of dummy rows, since all dummy rows
        // have the same work vars (the first block's prev_hash).
        Sha2BlockHasherFillerHelper::<C>::generate_carry_ae(
            first_dummy_row_cols_const.clone(),
            &mut first_dummy_row_cols_mut,
        );

        first_dummy_row_cols_const
    }
}

impl<F, C: Sha2BlockHasherVmConfig> Sha2BlockHasherChip<F, C> {
    #[allow(clippy::too_many_arguments)]
    fn fill_block_trace(
        &self,
        block_slice: &mut [F],
        input: &[u8],
        global_block_idx: usize, // 1-indexed
        prev_hash: &[C::Word],
        next_block_prev_hash: &[C::Word],
        request_id: usize,
    ) where
        F: PrimeField32,
    {
        debug_assert_eq!(input.len(), C::BLOCK_U8S);
        debug_assert_eq!(prev_hash.len(), C::HASH_WORDS);

        // Set request_id
        block_slice
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
            .for_each(|row_slice| {
                // Set request_id
                let cols = Sha2BlockHasherRoundColsRefMut::<F>::from::<C>(
                    &mut row_slice[..C::BLOCK_HASHER_WIDTH],
                );
                *cols.request_id = F::from_usize(request_id);
            });

        let input_words = (0..C::BLOCK_WORDS)
            .map(|i| {
                input[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                    .iter()
                    .fold(C::Word::from(0), |word, &byte| {
                        (word << 8) | u32::from(byte).into()
                    })
            })
            .collect::<Vec<_>>();

        // Fill in the inner trace
        self.inner.generate_block_trace(
            block_slice,
            C::BLOCK_HASHER_WIDTH,
            INNER_OFFSET,
            &input_words,
            self.bitwise_lookup_chip.clone(),
            &self.range_checker_chip,
            prev_hash,
            next_block_prev_hash,
            global_block_idx as u32,
        );
    }
}
