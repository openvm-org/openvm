use core::convert::TryInto;
use std::{
    borrow::BorrowMut,
    mem::{align_of, size_of},
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};
use p3_keccak_air::generate_trace_rows;

impl<F: PrimeField32> TraceFiller<F> for KeccakfVmFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let (trace, dummy_trace) = trace_matrix
            .values
            .split_at_mut(rows_used * NUM_KECCAKF_VM_COLS);

        let p3_dummy_trace: RowMajorMatrix<F> =
            generate_trace_rows(vec![[0u64; KECCAK_WIDTH_U64_LIMBS]; 1], 0);

        dummy_trace
            // Each Keccak-f round corresponds to exactly one trace row of width
            // NUM_KECCAKF_VM_COLS. We already reserved NUM_ROUNDS rows above
            // (NUM_ROUNDS * NUM_KECCAKF_VM_COLS elements).
            .chunks_exact_mut(NUM_KECCAKF_VM_COLS)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                let idx = row_idx % NUM_ROUNDS;
                row_slice[..NUM_KECCAK_PERM_COLS].copy_from_slice(
                    &p3_dummy_trace.values
                        [idx * NUM_KECCAK_PERM_COLS..(idx + 1) * NUM_KECCAK_PERM_COLS],
                );
                // Need to get rid of the accidental garbage data that might overflow
                // the F's prime field. Unfortunately, there
                // is no good way around this
                // SAFETY:
                // - row has exactly NUM_KECCAK_VM_COLS elements
                // - NUM_KECCAK_PERM_COLS offset is less than NUM_KECCAK_VM_COLS by design
                // - We're zeroing the remaining (NUM_KECCAK_VM_COLS - NUM_KECCAK_PERM_COLS)
                //   elements to clear any garbage data that might overflow the field
                unsafe {
                    std::ptr::write_bytes(
                        row_slice.as_mut_ptr().add(NUM_KECCAK_PERM_COLS) as *mut u8,
                        0,
                        (NUM_KECCAKF_VM_COLS - NUM_KECCAK_PERM_COLS) * size_of::<F>(),
                    );
                }
            });

        trace
            // Each Keccak-f round corresponds to exactly one trace row of width
            // NUM_KECCAKF_VM_COLS. We already reserved NUM_ROUNDS rows above
            // (NUM_ROUNDS * NUM_KECCAKF_VM_COLS elements).
            .chunks_exact_mut(NUM_KECCAKF_VM_COLS * NUM_ROUNDS)
            .for_each(|mut round_slice| {
                // each round takes up one row in the trace matrix
                // Safety: the initial prefix of the buffer of size NUM_KECCAKF_VM_COLS * NUM_ROUNDS
                // holds the record header
                let record: KeccakfVmRecordMut = unsafe {
                    get_record_from_slice(
                        &mut round_slice,
                        KeccakfVmRecordLayout {
                            metadata: KeccakfVmMetadata {},
                        },
                    )
                };
                let record = record.inner.clone();
                let mut timestamp = record.timestamp;

                // compute u64 preimage and postimage to not have to recompute per row
                let preimage_buffer_bytes = record.preimage_buffer_bytes;
                let mut preimage_buffer_bytes_u64: [u64; KECCAK_WIDTH_U64_LIMBS] =
                    [0; KECCAK_WIDTH_U64_LIMBS];
                for idx in 0..KECCAK_WIDTH_U64_LIMBS {
                    let le_bytes: [u8; 8] = preimage_buffer_bytes[8 * idx..8 * idx + 8]
                        .try_into()
                        .unwrap();
                    preimage_buffer_bytes_u64[idx] = u64::from_le_bytes(le_bytes);
                }

                let mut preimage_buffer_bytes_u64_transpose: [u64; KECCAK_WIDTH_U64_LIMBS] =
                    [0; KECCAK_WIDTH_U64_LIMBS];
                for y in 0..5 {
                    for x in 0..5 {
                        preimage_buffer_bytes_u64_transpose[x + 5 * y] =
                            preimage_buffer_bytes_u64[y + 5 * x];
                    }
                }

                let mut postimage_buffer_bytes_u64 = preimage_buffer_bytes_u64;
                tiny_keccak::keccakf(&mut postimage_buffer_bytes_u64);

                let mut postimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES] =
                    [0u8; KECCAK_WIDTH_BYTES];
                for idx in 0..KECCAK_WIDTH_U64_LIMBS {
                    let chunk: [u8; 8] = postimage_buffer_bytes_u64[idx].to_le_bytes();
                    postimage_buffer_bytes[8 * idx..8 * idx + 8].copy_from_slice(&chunk);
                }

                // fills in inner
                // the reason we give the transpose instead is inside, plonky3 transpose the
                // input so transpose of transpose fixes it
                let p3_trace: RowMajorMatrix<F> =
                    generate_trace_rows(vec![preimage_buffer_bytes_u64_transpose], 0);

                round_slice
                    .chunks_exact_mut(NUM_KECCAKF_VM_COLS)
                    .enumerate()
                    .for_each(|(row_idx, row)| {
                        row[..NUM_KECCAK_PERM_COLS].copy_from_slice(
                            &p3_trace.values[row_idx * NUM_KECCAK_PERM_COLS
                                ..(row_idx + 1) * NUM_KECCAK_PERM_COLS],
                        );

                        // fills in preimage_state_hi
                        let cols: &mut KeccakfOpCols<F> = row.borrow_mut();
                        for idx in 0..KECCAK_WIDTH_BYTES / 2 {
                            cols.preimage_state_hi[idx] =
                                F::from_canonical_u8(preimage_buffer_bytes[2 * idx + 1]);
                        }
                        // fills in postimage_state_hi
                        for idx in 0..KECCAK_WIDTH_BYTES / 2 {
                            cols.postimage_state_hi[idx] =
                                F::from_canonical_u8(postimage_buffer_bytes[2 * idx + 1]);
                        }
                        // fills in instruction
                        cols.instruction.pc = F::from_canonical_u32(record.pc);
                        cols.instruction.is_enabled = F::ONE;
                        cols.timestamp = F::from_canonical_u32(timestamp);
                        cols.instruction.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                        cols.instruction.buffer_ptr = F::from_canonical_u32(record.buffer);
                        cols.instruction.buffer_ptr_limbs =
                            record.buffer.to_le_bytes().map(F::from_canonical_u8);

                        let is_first_round = cols.inner.step_flags[0];
                        let is_final_round = cols.inner.step_flags[NUM_ROUNDS - 1];
                        cols.is_enabled_is_first_round = is_first_round;
                        cols.is_enabled_is_final_round = is_final_round;

                        // fills in memory offline checker
                        if row_idx == 0 {
                            mem_helper.fill(
                                record.register_aux_cols[0].prev_timestamp,
                                timestamp,
                                cols.mem_oc.register_aux_cols[0].as_mut(),
                            );
                            timestamp += 1;
                            for t in 0..KECCAK_WIDTH_U32_LIMBS {
                                mem_helper.fill(
                                    record.buffer_read_aux_cols[t].prev_timestamp,
                                    timestamp,
                                    cols.mem_oc.buffer_bytes_read_aux_cols[t].as_mut(),
                                );
                                timestamp += 1;
                            }

                            // safety: the following approach only works when self.pointer_max_bits
                            // >= 24
                            let limb_shift = 1
                                << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS
                                    - self.pointer_max_bits);
                            let buffer_limbs = record.buffer.to_le_bytes();
                            let need_range_check =
                                [buffer_limbs.last().unwrap(), buffer_limbs.last().unwrap()];
                            for pair in need_range_check.chunks_exact(2) {
                                self.bitwise_lookup_chip.request_range(
                                    (pair[0] * limb_shift) as u32,
                                    (pair[1] * limb_shift) as u32,
                                );
                            }
                        }

                        if row_idx == NUM_ROUNDS - 1 {
                            for t in 0..KECCAK_WIDTH_U32_LIMBS {
                                mem_helper.fill(
                                    record.buffer_write_aux_cols[t].prev_timestamp,
                                    timestamp,
                                    cols.mem_oc.buffer_bytes_write_aux_cols[t].as_mut(),
                                );
                                cols.mem_oc.buffer_bytes_write_aux_cols[t].prev_data = record
                                    .buffer_write_aux_cols[t]
                                    .prev_data
                                    .map(F::from_canonical_u8);
                                timestamp += 1;
                            }
                        }
                    });
            });
    }
}
