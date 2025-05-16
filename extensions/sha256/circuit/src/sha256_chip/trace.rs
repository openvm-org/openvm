use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{Result, TraceStep, VmStateMut},
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};

use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{memory_read, tracing_read, tracing_write};
use openvm_sha256_air::{
    get_flag_pt_array, u32_into_u16s, Sha256StepHelper, SHA256_BLOCK_BITS, SHA256_BLOCK_WORDS,
    SHA256_H, SHA256_ROWS_PER_BLOCK, SHA256_WORD_U8S,
};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_maybe_rayon::prelude::*};

use super::{
    Sha256VmDigestCols, Sha256VmRoundCols, Sha256VmStep, SHA256VM_CONTROL_WIDTH,
    SHA256VM_DIGEST_WIDTH,
};
use crate::{
    sha256_chip::{PaddingFlags, SHA256_READ_SIZE},
    SHA256VM_ROUND_WIDTH, SHA256_BLOCK_CELLS,
};

impl<F: PrimeField32, CTX> TraceStep<F, CTX> for Sha256VmStep {
    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        debug_assert_eq!(*opcode, Rv32Sha256Opcode::SHA256.global_opcode());
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        let trace = &mut trace[*trace_offset..];
        // Doing an untraced read to get the length to get the correct places to store the aux data
        let len = u32::from_le_bytes(memory_read(state.memory.data(), d, c.as_canonical_u32()));
        // need to pad with one 1 bit, 64 bits for the message length and then pad until the length
        // is divisible by [SHA256_BLOCK_BITS]
        let num_blocks = ((len << 3) as usize + 1 + 64).div_ceil(SHA256_BLOCK_BITS);

        let last_row_offset = (num_blocks * SHA256_ROWS_PER_BLOCK - 1) * width;
        let (dst, mut src) = {
            let last_digest_row: &mut Sha256VmDigestCols<F> =
                trace[last_row_offset..last_row_offset + SHA256VM_DIGEST_WIDTH].borrow_mut();

            last_digest_row.from_state.timestamp = F::from_canonical_u32(state.memory.timestamp());
            last_digest_row.from_state.pc = F::from_canonical_u32(*state.pc);
            let dst = tracing_read(
                state.memory,
                d,
                a.as_canonical_u32(),
                &mut last_digest_row.register_reads_aux[0],
            );
            let src = tracing_read(
                state.memory,
                d,
                b.as_canonical_u32(),
                &mut last_digest_row.register_reads_aux[1],
            );
            let len = tracing_read::<_, RV32_REGISTER_NUM_LIMBS>(
                state.memory,
                d,
                c.as_canonical_u32(),
                &mut last_digest_row.register_reads_aux[2],
            );

            last_digest_row.rd_ptr = *a;
            last_digest_row.rs1_ptr = *b;
            last_digest_row.rs2_ptr = *c;
            last_digest_row.dst_ptr = dst.map(F::from_canonical_u8);
            last_digest_row.src_ptr = src.map(F::from_canonical_u8);
            last_digest_row.len_data = len.map(F::from_canonical_u8);
            (u32::from_le_bytes(dst), u32::from_le_bytes(src))
        };

        // we will read [num_blocks] * [SHA256_BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(
            src as usize + num_blocks * SHA256_BLOCK_CELLS <= (1 << self.pointer_max_bits)
        );

        // // We can deduce the global block index from the trace offset
        // // Note: global block index is 1-based
        // let global_idx = *trace_offset / (SHA256_ROWS_PER_BLOCK * width) + 1;
        let mut prev_hash = SHA256_H;
        trace
            .chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .enumerate()
            .take(num_blocks)
            .for_each(|(block_idx, block_slice)| {
                let is_last_block = block_idx == num_blocks - 1;
                let mut read_data = [[0u8; SHA256_READ_SIZE]; 4];
                block_slice
                    .chunks_mut(width)
                    .enumerate()
                    .take(4)
                    .for_each(|(row_idx, row)| {
                        let cols: &mut Sha256VmRoundCols<F> =
                            row[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        read_data[row_idx] = tracing_read::<_, SHA256_READ_SIZE>(
                            state.memory,
                            e,
                            src,
                            &mut cols.read_aux,
                        );
                        cols.inner
                            .message_schedule
                            .carry_or_buffer
                            .iter_mut()
                            .zip(
                                read_data[row_idx]
                                    .map(F::from_canonical_u8)
                                    .chunks_exact(SHA256_WORD_U8S),
                            )
                            .for_each(|(buffer, data)| {
                                buffer.copy_from_slice(data);
                            });
                        src += SHA256_READ_SIZE as u32;
                    });

                let digest_row = &mut block_slice[(SHA256_ROWS_PER_BLOCK - 1) * width..];
                let digest_cols: &mut Sha256VmDigestCols<F> =
                    digest_row[..SHA256VM_DIGEST_WIDTH].borrow_mut();
                digest_cols.inner.prev_hash =
                    prev_hash.map(|x| u32_into_u16s(x).map(F::from_canonical_u32));
                digest_cols.inner.flags.local_block_idx = F::from_canonical_usize(block_idx);
                digest_cols.inner.flags.is_last_block = F::from_bool(is_last_block);
                digest_cols.control.len = F::from_canonical_u32(len);
                digest_cols.control.read_ptr = F::from_canonical_u32(src);
                digest_cols.control.cur_timestamp = F::from_canonical_u32(state.memory.timestamp());
                let padded_input = get_padded_input(
                    read_data.concat().try_into().unwrap(),
                    len,
                    block_idx,
                    is_last_block,
                );
                Sha256StepHelper::get_block_hash(&mut prev_hash, padded_input);
            });

        let last_digest_row: &mut Sha256VmDigestCols<F> =
            trace[last_row_offset..last_row_offset + SHA256VM_DIGEST_WIDTH].borrow_mut();
        tracing_write(
            state.memory,
            e,
            dst,
            &prev_hash
                .map(|x| x.to_be_bytes())
                .concat()
                .try_into()
                .unwrap(),
            &mut last_digest_row.writes_aux,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        *trace_offset += num_blocks * SHA256_ROWS_PER_BLOCK * width;
        Ok(())
    }

    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut [F],
        width: usize,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync,
    {
        let mem_ptr_shift: u32 =
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits);

        // During the first pass we will fill out most of the matrix
        // But there are some cells that can't be generated by the first pass so we will do a second
        // pass over the matrix
        trace
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .enumerate()
            .for_each(|(block_idx, block_slice)| {
                if block_idx * SHA256_ROWS_PER_BLOCK >= rows_used {
                    // Fill in the invalid rows
                    block_slice.par_chunks_mut(width).for_each(|row| {
                        let cols: &mut Sha256VmRoundCols<F> =
                            row[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        self.inner.generate_default_row(&mut cols.inner);
                    });
                    return;
                }

                // The read data is kept in the buffer of the first 4 round cols
                let read_data: [u8; SHA256_BLOCK_CELLS] = block_slice
                    .chunks_exact(width)
                    .take(4)
                    .map(|row| {
                        let cols: &Sha256VmRoundCols<F> = row[..SHA256VM_ROUND_WIDTH].borrow();
                        cols.inner.message_schedule.carry_or_buffer.as_flattened()
                    })
                    .flatten()
                    .map(|x| x.as_canonical_u32() as u8)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let digest_offset = width * (SHA256_ROWS_PER_BLOCK - 1);
                let (local_block_idx, len, is_last_block, prev_hash) = {
                    let digest_cols: &mut Sha256VmDigestCols<F> = block_slice
                        [digest_offset..digest_offset + SHA256VM_DIGEST_WIDTH]
                        .borrow_mut();
                    (
                        digest_cols.inner.flags.local_block_idx.as_canonical_u32() as usize,
                        digest_cols.control.len.as_canonical_u32(),
                        digest_cols.inner.flags.is_last_block.is_one(),
                        digest_cols
                            .inner
                            .prev_hash
                            .map(|x| x[0].as_canonical_u32() + (x[1].as_canonical_u32() << 16)),
                    )
                };
                let mut has_padding_occurred = local_block_idx * SHA256_BLOCK_CELLS > len as usize;
                let message_left = if has_padding_occurred {
                    0
                } else {
                    len as usize - local_block_idx * SHA256_BLOCK_CELLS
                };

                let padded_input = get_padded_input(read_data, len, local_block_idx, is_last_block);
                let padded_input: [u32; SHA256_BLOCK_WORDS] = array::from_fn(|j| {
                    u32::from_be_bytes(
                        padded_input[j * SHA256_WORD_U8S..(j + 1) * SHA256_WORD_U8S]
                            .try_into()
                            .unwrap(),
                    )
                });

                self.inner.generate_block_trace::<F>(
                    block_slice,
                    width,
                    SHA256VM_CONTROL_WIDTH,
                    &padded_input,
                    self.bitwise_lookup_chip.as_ref(),
                    &prev_hash,
                    is_last_block,
                    block_idx as u32 + 1, // global block index is 1-based
                    local_block_idx as u32,
                );

                let (round_rows, digest_row) = block_slice.split_at_mut(digest_offset);
                let digest_cols: &mut Sha256VmDigestCols<F> =
                    digest_row[..SHA256VM_DIGEST_WIDTH].borrow_mut();
                let len = digest_cols.control.len;
                let read_ptr = digest_cols.control.read_ptr;
                let timestamp = digest_cols.control.cur_timestamp;

                // Fill in the first 4 round rows
                round_rows
                    .chunks_mut(width)
                    .take(4)
                    .enumerate()
                    .for_each(|(row, row_slice)| {
                        let cols: &mut Sha256VmRoundCols<F> =
                            row_slice[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        cols.control.len = len;
                        cols.control.read_ptr =
                            read_ptr - F::from_canonical_usize(SHA256_READ_SIZE * (4 - row));
                        cols.control.cur_timestamp = timestamp - F::from_canonical_usize(4 - row);
                        mem_helper.fill_from_prev(
                            cols.control.cur_timestamp.as_canonical_u32(),
                            cols.read_aux.as_mut(),
                        );
                        if (row + 1) * SHA256_READ_SIZE <= message_left {
                            cols.control.pad_flags = get_flag_pt_array(
                                &self.padding_encoder,
                                PaddingFlags::NotPadding as usize,
                            )
                            .map(F::from_canonical_u32);
                        } else if !has_padding_occurred {
                            has_padding_occurred = true;
                            let len = message_left - row * SHA256_READ_SIZE;
                            cols.control.pad_flags = get_flag_pt_array(
                                &self.padding_encoder,
                                if row == 3 && is_last_block {
                                    PaddingFlags::FirstPadding0_LastRow
                                } else {
                                    PaddingFlags::FirstPadding0
                                } as usize
                                    + len,
                            )
                            .map(F::from_canonical_u32);
                        } else {
                            cols.control.pad_flags = get_flag_pt_array(
                                &self.padding_encoder,
                                if row == 3 && is_last_block {
                                    PaddingFlags::EntirePaddingLastRow
                                } else {
                                    PaddingFlags::EntirePadding
                                } as usize,
                            )
                            .map(F::from_canonical_u32);
                        }
                        cols.control.padding_occurred = F::from_bool(has_padding_occurred);
                    });

                // Fill in the remaining round rows

                round_rows
                    .par_chunks_mut(width)
                    .skip(4)
                    .for_each(|row_slice| {
                        let cols: &mut Sha256VmRoundCols<F> =
                            row_slice[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        cols.control.len = len;
                        cols.control.read_ptr = read_ptr;
                        cols.control.cur_timestamp = timestamp;
                        cols.control.pad_flags = get_flag_pt_array(
                            &self.padding_encoder,
                            PaddingFlags::NotConsidered as usize,
                        )
                        .map(F::from_canonical_u32);
                        cols.control.padding_occurred = F::from_bool(has_padding_occurred);
                    });

                // Fill in the digest row
                if is_last_block {
                    has_padding_occurred = false;
                }
                digest_cols.control.pad_flags =
                    get_flag_pt_array(&self.padding_encoder, PaddingFlags::NotConsidered as usize)
                        .map(F::from_canonical_u32);
                if is_last_block {
                    let mut timestamp = digest_cols.from_state.timestamp.as_canonical_u32();
                    digest_cols.register_reads_aux.iter_mut().for_each(|aux| {
                        mem_helper.fill_from_prev(timestamp, aux.as_mut());
                        timestamp += 1;
                    });
                    mem_helper.fill_from_prev(
                        digest_cols.control.cur_timestamp.as_canonical_u32(),
                        digest_cols.writes_aux.as_mut(),
                    );
                    self.bitwise_lookup_chip.request_range(
                        digest_cols.dst_ptr[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
                            * mem_ptr_shift,
                        digest_cols.src_ptr[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
                            * mem_ptr_shift,
                    );
                }
                digest_cols.control.padding_occurred = F::from_bool(has_padding_occurred);
            });

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        trace[width..]
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .take(rows_used / SHA256_ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.inner
                    .generate_missing_cells(chunk, width, SHA256VM_CONTROL_WIDTH);
            });
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32Sha256Opcode::SHA256)
    }
}

fn get_padded_input(
    block_input: [u8; SHA256_BLOCK_CELLS],
    message_len: u32,
    local_block_idx: usize,
    is_last_block: bool,
) -> [u8; SHA256_BLOCK_CELLS] {
    let has_padding_occurred = local_block_idx * SHA256_BLOCK_CELLS > message_len as usize;
    let message_left = if has_padding_occurred {
        0
    } else {
        message_len as usize - local_block_idx * SHA256_BLOCK_CELLS
    };

    array::from_fn(|j| {
        if j < message_left {
            block_input[j]
        } else if j == message_left && !has_padding_occurred {
            1 << (RV32_CELL_BITS - 1)
        } else if !is_last_block || j < SHA256_BLOCK_CELLS - 4 {
            0u8
        } else {
            let shift_amount = (SHA256_BLOCK_CELLS - j - 1) * RV32_CELL_BITS;
            ((message_len * RV32_CELL_BITS as u32)
                .checked_shr(shift_amount as u32)
                .unwrap_or(0)
                & ((1 << RV32_CELL_BITS) - 1)) as u8
        }
    })
}
