use std::{slice, sync::Arc};

use openvm_circuit::arch::get_record_from_slice;
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_sha2_air::{
    be_limbs_into_word, le_limbs_into_word, Sha2BlockHasherFillerHelper, Sha2RoundColsRef,
    Sha2RoundColsRefMut,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip,
};

use crate::{
    Sha2BlockHasherChip, Sha2BlockHasherRoundColsRefMut, Sha2BlockHasherVmConfig, Sha2Config,
    Sha2Metadata, Sha2RecordLayout, Sha2RecordMut, Sha2SharedRecords, INNER_OFFSET,
};

// We don't use the record arena associated with this chip. Instead, we will use the record arena
// provided by the main chip, which will be passed to this chip after the main chip's tracegen is
// done.
impl<R, SC, C: Sha2Config> Chip<R, CpuBackend<SC>> for Sha2BlockHasherChip<Val<SC>, C>
where
    Val<SC>: PrimeField32,
    SC: StarkGenericConfig,
{
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        // SAFETY: the tracegen for Sha2MainChip must be done before this chip's tracegen
        let mut records = self.records.lock().unwrap();
        let mut records = records.take().unwrap();
        let rows_used = records.num_records * C::ROWS_PER_BLOCK;

        let height = next_power_of_two_or_zero(rows_used);
        let trace = Val::<SC>::zero_vec(height * C::BLOCK_HASHER_WIDTH);
        let mut trace_matrix = RowMajorMatrix::new(trace, C::BLOCK_HASHER_WIDTH);

        self.fill_trace(&mut trace_matrix, &mut records, rows_used);

        let mut cols_first_row = super::Sha2BlockHasherRoundColsRefMut::<Val<SC>>::from::<C>(
            &mut trace_matrix.values[..C::BLOCK_HASHER_WIDTH],
        );
        // cols_first_row.inner.schedule_helper.intermed_12.row_mut(3)[1] = Val::<SC>::ONE;

        AirProvingContext::simple(Arc::new(trace_matrix), vec![])
    }
}

impl<F, C> Sha2BlockHasherChip<F, C>
where
    F: PrimeField32,
    C: Sha2BlockHasherVmConfig,
{
    fn fill_trace(
        &self,
        trace_matrix: &mut RowMajorMatrix<F>,
        records: &mut Sha2SharedRecords<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let trace = &mut trace_matrix.values[..];

        // grab all the records
        // we need to do this first, so we can pass (this_block.prev_hash, next_block.prev_hash) to
        // each block (in the call to fill_block_trace)
        let (records, prev_hashes): (Vec<_>, Vec<_>) = records
            .matrix
            .par_rows_mut()
            .take(records.num_records)
            .map(|mut record| {
                // SAFETY:
                // - caller ensures `records` contains a valid record representation that was
                //   previously written by the executor
                // - records contains a valid Sha2RecordMut with the exact layout specified
                // - get_record_from_slice will correctly split the buffer into header, input, and
                //   aux components based on this layout
                let record: Sha2RecordMut = unsafe {
                    get_record_from_slice(
                        &mut record,
                        Sha2RecordLayout {
                            metadata: Sha2Metadata {
                                variant: C::VARIANT,
                            },
                        },
                    )
                };

                let prev_hash = (0..C::HASH_WORDS)
                    .map(|i| {
                        le_limbs_into_word::<C>(
                            &record.prev_state[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                                .iter()
                                .map(|x| *x as u32)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();

                (record, prev_hash)
            })
            .unzip();

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
                records
                    .par_iter()
                    .zip(prev_hashes_and_next_block_prev_hashes),
            )
            .enumerate()
            .for_each(
                |(block_idx, (block_slice, (record, (prev_hash, next_block_prev_hash))))| {
                    self.fill_block_trace(
                        block_slice,
                        record.message_bytes,
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
        let first_dummy_row_cols_const = self.fill_first_dummy_row(
            &mut trace[rows_used * C::BLOCK_HASHER_WIDTH..(rows_used + 1) * C::BLOCK_HASHER_WIDTH],
            &prev_hashes[0],
        );

        // fill in the rest of the dummy rows
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
    }

    fn fill_first_dummy_row(
        &self,
        first_dummy_row_mut: &mut [F],
        first_block_prev_hash: &[C::Word],
    ) -> Sha2RoundColsRef<F> {
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
        // work vars and row_idx)
        self.inner.generate_default_row(
            &mut first_dummy_row_cols_mut,
            first_block_prev_hash,
            None,
            None,
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
                *cols.request_id = F::from_canonical_usize(request_id);
            });

        let input_words = (0..C::BLOCK_WORDS)
            .map(|i| {
                be_limbs_into_word::<C>(
                    &input[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        // Fill in the inner trace
        self.inner.generate_block_trace(
            block_slice,
            C::BLOCK_HASHER_WIDTH,
            INNER_OFFSET,
            &input_words,
            self.bitwise_lookup_chip.clone(),
            prev_hash,
            next_block_prev_hash,
            global_block_idx as u32,
        );
    }
}
