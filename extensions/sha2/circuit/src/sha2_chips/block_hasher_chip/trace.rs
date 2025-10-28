use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
    cmp::min,
    marker::PhantomData,
    mem,
    ops::Range,
    sync::Arc,
};

use itertools::Itertools;
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
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    encoder::Encoder,
    utils::{compose, next_power_of_two_or_zero},
    AlignedBytesBorrow,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_sha2_air::{
    be_limbs_into_word, big_sig0, big_sig1, ch, le_limbs_into_word, maj,
    set_arrayview_from_u8_slice, word_into_bits, word_into_u16_limbs, Sha2BlockHasherFillerHelper,
    Sha2DigestColsRefMut, Sha2RoundColsRef, Sha2RoundColsRefMut, WrappingAdd,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};

use crate::{
    Sha2BlockHasherChip, Sha2BlockHasherVmConfig, Sha2BlockHasherVmRoundColsRefMut, Sha2Config,
    Sha2Metadata, Sha2RecordLayout, Sha2RecordMut, INNER_OFFSET,
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
        // 1 record per instruction
        let num_instructions = records.height();
        let rows_used = num_instructions * C::ROWS_PER_BLOCK;

        let height = next_power_of_two_or_zero(rows_used);
        let trace = Val::<SC>::zero_vec(height * C::BLOCK_HASHER_WIDTH);
        let mut trace_matrix = RowMajorMatrix::new(trace, C::BLOCK_HASHER_WIDTH);

        self.fill_trace(&mut trace_matrix, rows_used, &mut records);

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
        rows_used: usize,
        records: &mut RowMajorMatrix<F>,
    ) {
        if rows_used == 0 {
            return;
        }

        let trace = &mut trace_matrix.values[..];

        // fill in dummy rows
        trace[rows_used * C::BLOCK_HASHER_WIDTH..]
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
            .for_each(|row| {
                let cols = Sha2RoundColsRefMut::from::<C>(
                    &mut row[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
                );

                self.inner.generate_default_row(cols);
            });

        // fill in used rows
        trace[..rows_used * C::BLOCK_HASHER_WIDTH]
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH * C::ROWS_PER_BLOCK)
            .zip(records.par_rows_mut())
            .enumerate()
            .for_each(|(block_idx, (block_slice, mut record))| {
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
                        be_limbs_into_word::<C>(
                            &record.prev_state[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                                .iter()
                                .map(|x| *x as u32)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();

                self.fill_block_trace(
                    block_slice,
                    record.message_bytes,
                    block_idx + 1, // 1-indexed
                    &prev_hash,
                    block_idx,
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

        {
            let mut rows_24_and_25 = trace
                .chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
                .skip(24)
                .take(2)
                .collect::<Vec<_>>();
            let row_24 = rows_24_and_25.remove(0);
            let row_25 = rows_24_and_25.remove(0);
            let row_24_cols = Sha2RoundColsRef::from::<C>(
                &row_24[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            let mut row_25_cols = Sha2RoundColsRefMut::from::<C>(
                &mut row_25[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );

            // Sha2BlockHasherFillerHelper::<C>::generate_carry_ae(row_24_cols, &mut row_25_cols);

            println!("row 25 carry_a: {:?}", row_25_cols.work_vars.carry_a);

            println!("row 25 a: {:?}", row_25_cols.work_vars.a);
            println!("row 25 e: {:?}", row_25_cols.work_vars.e);
        }

        {
            let mut rows = trace
                .chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
                .collect::<Vec<_>>();
            let row_0 = rows.remove(0);
            let row_31 = rows.remove(30);
            let row_31_cols = Sha2RoundColsRef::from::<C>(
                &row_31[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );
            let mut row_0_cols = Sha2RoundColsRefMut::from::<C>(
                &mut row_0[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH],
            );

            // Sha2BlockHasherFillerHelper::<C>::generate_carry_ae(row_31_cols, &mut row_0_cols);

            println!("row 0 carry_a: {:?}", row_0_cols.work_vars.carry_a);

            println!("row 0 a: {:?}", row_0_cols.work_vars.a);
            println!("row 0 e: {:?}", row_0_cols.work_vars.e);
        }
    }
}

impl<F, C: Sha2BlockHasherVmConfig> Sha2BlockHasherChip<F, C> {
    #[allow(clippy::too_many_arguments)]
    fn fill_block_trace(
        &self,
        block_slice: &mut [F],
        input: &[u8],
        global_block_idx: usize,
        prev_hash: &[C::Word],
        request_id: usize,
    ) where
        F: PrimeField32,
    {
        debug_assert_eq!(input.len(), C::BLOCK_U8S);
        debug_assert_eq!(prev_hash.len(), C::HASH_WORDS);

        // Set request_id and fill the input into carry_or_buffer
        block_slice
            .par_chunks_exact_mut(C::BLOCK_HASHER_WIDTH)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                // Set request_id
                let cols = Sha2BlockHasherVmRoundColsRefMut::<F>::from::<C>(
                    &mut row_slice[..C::BLOCK_HASHER_WIDTH],
                );
                *cols.request_id = F::from_canonical_usize(request_id);

                // Fill the input into carry_or_buffer
                if row_idx < C::MESSAGE_ROWS {
                    let mut round_cols = Sha2RoundColsRefMut::<F>::from::<C>(
                        row_slice[INNER_OFFSET..INNER_OFFSET + C::SUBAIR_ROUND_WIDTH].borrow_mut(),
                    );
                    // We don't actually need to set carry_or_buffer for the first 4 rows, because
                    // the subair won't actually use it (we used to store the memory read into here
                    // in the old SHA-2 chip). We will remove the subair constraints that constrain
                    // carry_or_buffer for the first 4 rows in the future. Then, we can skip setting
                    // carry_or_buffer for the first 4 rows here.
                    // TODO: resolve this before getting this PR reviewed.
                    set_arrayview_from_u8_slice(
                        &mut round_cols.message_schedule.carry_or_buffer,
                        input[row_idx * C::ROUNDS_PER_ROW * C::WORD_U8S
                            ..(row_idx + 1) * C::ROUNDS_PER_ROW * C::WORD_U8S]
                            .iter()
                            .copied(),
                    );
                }
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
            true,
            global_block_idx as u32,
        );
    }
}
