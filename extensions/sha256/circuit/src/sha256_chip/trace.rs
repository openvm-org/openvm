use std::{
    array,
    borrow::{Borrow, BorrowMut},
    cmp::min,
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
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_sha256_air::{
    get_flag_pt_array, get_sha256_num_blocks, Sha256FillerHelper, SHA256_BLOCK_BITS, SHA256_H,
    SHA256_ROWS_PER_BLOCK,
};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};

use super::{
    Sha256VmDigestCols, Sha256VmExecutor, Sha256VmRoundCols, SHA256VM_CONTROL_WIDTH,
    SHA256VM_DIGEST_WIDTH,
};
use crate::{
    sha256_chip::{PaddingFlags, SHA256_READ_SIZE, SHA256_REGISTER_READS, SHA256_WRITE_SIZE},
    sha256_solve, Sha256VmControlCols, Sha256VmFiller, SHA256VM_ROUND_WIDTH, SHA256VM_WIDTH,
    SHA256_BLOCK_CELLS, SHA256_MAX_MESSAGE_LEN, SHA256_NUM_READ_ROWS,
};

#[derive(Clone, Copy)]
pub struct Sha256VmMetadata {
    pub num_blocks: u32,
}

impl MultiRowMetadata for Sha256VmMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_blocks as usize * SHA256_ROWS_PER_BLOCK
    }
}

pub(crate) type Sha256VmRecordLayout = MultiRowLayout<Sha256VmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Sha256VmRecordHeader {
    pub from_pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub dst_ptr: u32,
    pub src_ptr: u32,
    pub len: u32,

    pub register_reads_aux: [MemoryReadAuxRecord; SHA256_REGISTER_READS],
    pub write_aux: MemoryWriteBytesAuxRecord<SHA256_WRITE_SIZE>,
}

pub struct Sha256VmRecordMut<'a> {
    pub inner: &'a mut Sha256VmRecordHeader,
    // Having a continuous slice of the input is useful for fast hashing in `execute`
    pub input: &'a mut [u8],
    pub read_aux: &'a mut [MemoryReadAuxRecord],
}

/// Custom borrowing that splits the buffer into a fixed `Sha256VmRecord` header
/// followed by a slice of `u8`'s of length `SHA256_BLOCK_CELLS * num_blocks` where `num_blocks` is
/// provided at runtime, followed by a slice of `MemoryReadAuxRecord`'s of length
/// `SHA256_NUM_READ_ROWS * num_blocks`. Uses `align_to_mut()` to make sure the slice is properly
/// aligned to `MemoryReadAuxRecord`. Has debug assertions that check the size and alignment of the
/// slices.
impl<'a> CustomBorrow<'a, Sha256VmRecordMut<'a>, Sha256VmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Sha256VmRecordLayout) -> Sha256VmRecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits and
        //   constants are guaranteed <= self.len() by layout precondition
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Sha256VmRecordHeader>()) };

        // SAFETY:
        // - layout guarantees rest has sufficient length for input data
        // - The layout size calculation includes num_blocks * SHA256_BLOCK_CELLS bytes after header
        // - num_blocks is derived from the message length ensuring correct sizing
        // - SHA256_BLOCK_CELLS is a compile-time constant (64 bytes per block)
        let (input, rest) = unsafe {
            rest.split_at_mut_unchecked((layout.metadata.num_blocks as usize) * SHA256_BLOCK_CELLS)
        };

        // SAFETY:
        // - rest is a valid mutable slice from the previous split
        // - align_to_mut guarantees the middle slice is properly aligned for MemoryReadAuxRecord
        // - The subslice operation [..num_blocks * SHA256_NUM_READ_ROWS] validates sufficient
        //   capacity
        // - Layout calculation ensures space for alignment padding plus required aux records
        let (_, read_aux_buf, _) = unsafe { rest.align_to_mut::<MemoryReadAuxRecord>() };
        Sha256VmRecordMut {
            inner: header_buf.borrow_mut(),
            input,
            read_aux: &mut read_aux_buf
                [..(layout.metadata.num_blocks as usize) * SHA256_NUM_READ_ROWS],
        }
    }

    unsafe fn extract_layout(&self) -> Sha256VmRecordLayout {
        let header: &Sha256VmRecordHeader = self.borrow();
        Sha256VmRecordLayout {
            metadata: Sha256VmMetadata {
                num_blocks: get_sha256_num_blocks(header.len),
            },
        }
    }
}

impl SizedRecord<Sha256VmRecordLayout> for Sha256VmRecordMut<'_> {
    fn size(layout: &Sha256VmRecordLayout) -> usize {
        let mut total_len = size_of::<Sha256VmRecordHeader>();
        total_len += layout.metadata.num_blocks as usize * SHA256_BLOCK_CELLS;
        // Align the pointer to the alignment of `MemoryReadAuxRecord`
        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += layout.metadata.num_blocks as usize
            * SHA256_NUM_READ_ROWS
            * size_of::<MemoryReadAuxRecord>();
        total_len
    }

    fn alignment(_layout: &Sha256VmRecordLayout) -> usize {
        align_of::<Sha256VmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for Sha256VmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, Sha256VmRecordLayout, Sha256VmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32Sha256Opcode::SHA256)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(*opcode, Rv32Sha256Opcode::SHA256.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Reading the length first to allocate a record of correct size
        let len = read_rv32_register(state.memory.data(), c.as_canonical_u32());

        let num_blocks = get_sha256_num_blocks(len);
        let record = state.ctx.alloc(MultiRowLayout {
            metadata: Sha256VmMetadata { num_blocks },
        });

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

        record.inner.dst_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rd_ptr,
            &mut record.inner.register_reads_aux[0].prev_timestamp,
        ));
        record.inner.src_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs1_ptr,
            &mut record.inner.register_reads_aux[1].prev_timestamp,
        ));
        record.inner.len = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs2_ptr,
            &mut record.inner.register_reads_aux[2].prev_timestamp,
        ));

        // we will read [num_blocks] * [SHA256_BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(
            record.inner.src_ptr as usize + num_blocks as usize * SHA256_BLOCK_CELLS
                <= (1 << self.pointer_max_bits)
        );
        debug_assert!(
            record.inner.dst_ptr as usize + SHA256_WRITE_SIZE <= (1 << self.pointer_max_bits)
        );
        // We don't support messages longer than 2^29 bytes
        debug_assert!(record.inner.len < SHA256_MAX_MESSAGE_LEN as u32);

        for block_idx in 0..num_blocks as usize {
            // Reads happen on the first 4 rows of each block
            for row in 0..SHA256_NUM_READ_ROWS {
                let read_idx = block_idx * SHA256_NUM_READ_ROWS + row;
                let row_input: [u8; SHA256_READ_SIZE] = tracing_read(
                    state.memory,
                    RV32_MEMORY_AS,
                    record.inner.src_ptr + (read_idx * SHA256_READ_SIZE) as u32,
                    &mut record.read_aux[read_idx].prev_timestamp,
                );
                record.input[read_idx * SHA256_READ_SIZE..(read_idx + 1) * SHA256_READ_SIZE]
                    .copy_from_slice(&row_input);
            }
        }

        let output = sha256_solve(&record.input[..len as usize]);
        tracing_write(
            state.memory,
            RV32_MEMORY_AS,
            record.inner.dst_ptr,
            output,
            &mut record.inner.write_aux.prev_timestamp,
            &mut record.inner.write_aux.prev_data,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for Sha256VmFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let mut chunks = Vec::with_capacity(trace_matrix.height() / SHA256_ROWS_PER_BLOCK);
        let mut sizes = Vec::with_capacity(trace_matrix.height() / SHA256_ROWS_PER_BLOCK);
        let mut trace = &mut trace_matrix.values[..];
        let mut num_blocks_so_far = 0;

        // First pass over the trace to get the number of blocks for each instruction
        // and divide the matrix into chunks of needed sizes
        loop {
            if num_blocks_so_far * SHA256_ROWS_PER_BLOCK >= rows_used {
                // Push all the padding rows as a single chunk and break
                chunks.push(trace);
                sizes.push((0, num_blocks_so_far));
                break;
            } else {
                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - header is the first element of the record
                let record: &Sha256VmRecordHeader =
                    unsafe { get_record_from_slice(&mut trace, ()) };
                let num_blocks = ((record.len << 3) as usize + 1 + 64).div_ceil(SHA256_BLOCK_BITS);
                let (chunk, rest) =
                    trace.split_at_mut(SHA256VM_WIDTH * SHA256_ROWS_PER_BLOCK * num_blocks);
                chunks.push(chunk);
                sizes.push((num_blocks, num_blocks_so_far));
                num_blocks_so_far += num_blocks;
                trace = rest;
            }
        }

        // During the first pass we will fill out most of the matrix
        // But there are some cells that can't be generated by the first pass so we will do a second
        // pass over the matrix later
        chunks.par_iter_mut().zip(sizes.par_iter()).for_each(
            |(slice, (num_blocks, global_block_offset))| {
                if global_block_offset * SHA256_ROWS_PER_BLOCK >= rows_used {
                    // Fill in the invalid rows
                    slice.par_chunks_mut(SHA256VM_WIDTH).for_each(|row| {
                        // Need to get rid of the accidental garbage data that might overflow the
                        // F's prime field. Unfortunately, there is no good way around this
                        // SAFETY:
                        // - row has exactly SHA256VM_WIDTH elements
                        // - We're zeroing all SHA256VM_WIDTH elements to clear any garbage data
                        //   that might overflow the field
                        // - Casting F* to u8* preserves validity for write_bytes operation
                        // - SHA256VM_WIDTH * size_of::<F>() correctly calculates total bytes to
                        //   zero
                        unsafe {
                            std::ptr::write_bytes(
                                row.as_mut_ptr() as *mut u8,
                                0,
                                SHA256VM_WIDTH * size_of::<F>(),
                            );
                        }
                        let cols: &mut Sha256VmRoundCols<F> =
                            row[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        self.inner.generate_default_row(&mut cols.inner);
                    });
                    return;
                }

                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - slice contains a valid Sha256VmRecord with the exact layout specified
                // - get_record_from_slice will correctly split the buffer into header, input, and
                //   aux components based on this layout
                let record: Sha256VmRecordMut = unsafe {
                    get_record_from_slice(
                        slice,
                        Sha256VmRecordLayout {
                            metadata: Sha256VmMetadata {
                                num_blocks: *num_blocks as u32,
                            },
                        },
                    )
                };

                let mut input: Vec<u8> = Vec::with_capacity(SHA256_BLOCK_CELLS * num_blocks);
                input.extend_from_slice(record.input);
                let mut padded_input = input.clone();
                let len = record.inner.len as usize;
                let padded_input_len = padded_input.len();
                padded_input[len] = 1 << (RV32_CELL_BITS - 1);
                padded_input[len + 1..padded_input_len - 4].fill(0);
                padded_input[padded_input_len - 4..]
                    .copy_from_slice(&((len as u32) << 3).to_be_bytes());

                let mut prev_hashes = Vec::with_capacity(*num_blocks);
                prev_hashes.push(SHA256_H);
                for i in 0..*num_blocks - 1 {
                    prev_hashes.push(Sha256FillerHelper::get_block_hash(
                        &prev_hashes[i],
                        padded_input[i * SHA256_BLOCK_CELLS..(i + 1) * SHA256_BLOCK_CELLS]
                            .try_into()
                            .unwrap(),
                    ));
                }
                // Copy the read aux records and input to another place to safely fill in the trace
                // matrix without overwriting the record
                let mut read_aux_records = Vec::with_capacity(SHA256_NUM_READ_ROWS * num_blocks);
                read_aux_records.extend_from_slice(record.read_aux);
                let vm_record = record.inner.clone();

                slice
                    .par_chunks_exact_mut(SHA256VM_WIDTH * SHA256_ROWS_PER_BLOCK)
                    .enumerate()
                    .for_each(|(block_idx, block_slice)| {
                        // Need to get rid of the accidental garbage data that might overflow the
                        // F's prime field. Unfortunately, there is no good way around this
                        // SAFETY:
                        // - block_slice comes from par_chunks_exact_mut with exact size guarantee
                        // - Length is SHA256_ROWS_PER_BLOCK * SHA256VM_WIDTH * size_of::<F>() bytes
                        // - Zeroing entire blocks prevents using garbage data
                        // - The subsequent trace filling will overwrite with valid values
                        unsafe {
                            std::ptr::write_bytes(
                                block_slice.as_mut_ptr() as *mut u8,
                                0,
                                SHA256_ROWS_PER_BLOCK * SHA256VM_WIDTH * size_of::<F>(),
                            );
                        }
                        self.fill_block_trace::<F>(
                            block_slice,
                            &vm_record,
                            &read_aux_records[block_idx * SHA256_NUM_READ_ROWS
                                ..(block_idx + 1) * SHA256_NUM_READ_ROWS],
                            &input[block_idx * SHA256_BLOCK_CELLS
                                ..(block_idx + 1) * SHA256_BLOCK_CELLS],
                            &padded_input[block_idx * SHA256_BLOCK_CELLS
                                ..(block_idx + 1) * SHA256_BLOCK_CELLS],
                            block_idx == *num_blocks - 1,
                            *global_block_offset + block_idx,
                            block_idx,
                            prev_hashes[block_idx],
                            mem_helper,
                        );
                    });
            },
        );

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        trace_matrix.values[SHA256VM_WIDTH..]
            .par_chunks_mut(SHA256VM_WIDTH * SHA256_ROWS_PER_BLOCK)
            .take(rows_used / SHA256_ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.inner
                    .generate_missing_cells(chunk, SHA256VM_WIDTH, SHA256VM_CONTROL_WIDTH);
            });
    }
}

impl Sha256VmFiller {
    #[allow(clippy::too_many_arguments)]
    fn fill_block_trace<F: PrimeField32>(
        &self,
        block_slice: &mut [F],
        record: &Sha256VmRecordHeader,
        read_aux_records: &[MemoryReadAuxRecord],
        input: &[u8],
        padded_input: &[u8],
        is_last_block: bool,
        global_block_idx: usize,
        local_block_idx: usize,
        prev_hash: [u32; 8],
        mem_helper: &MemoryAuxColsFactory<F>,
    ) {
        debug_assert_eq!(input.len(), SHA256_BLOCK_CELLS);
        debug_assert_eq!(padded_input.len(), SHA256_BLOCK_CELLS);
        debug_assert_eq!(read_aux_records.len(), SHA256_NUM_READ_ROWS);

        let padded_input = array::from_fn(|i| {
            u32::from_be_bytes(padded_input[i * 4..(i + 1) * 4].try_into().unwrap())
        });

        let block_start_timestamp = record.timestamp
            + (SHA256_REGISTER_READS + SHA256_NUM_READ_ROWS * local_block_idx) as u32;

        let read_cells = (SHA256_BLOCK_CELLS * local_block_idx) as u32;
        let block_start_read_ptr = record.src_ptr + read_cells;

        let message_left = if record.len <= read_cells {
            0
        } else {
            (record.len - read_cells) as usize
        };

        // -1 means that padding occurred before the start of the block
        // 18 means that no padding occurred on this block
        let first_padding_row = if record.len < read_cells {
            -1
        } else if message_left < SHA256_BLOCK_CELLS {
            (message_left / SHA256_READ_SIZE) as i32
        } else {
            18
        };

        // Fill in the VM columns first because the inner `carry_or_buffer` needs to be filled in
        block_slice
            .par_chunks_exact_mut(SHA256VM_WIDTH)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                // Handle round rows and digest row separately
                if row_idx == SHA256_ROWS_PER_BLOCK - 1 {
                    // This is a digest row
                    let digest_cols: &mut Sha256VmDigestCols<F> =
                        row_slice[..SHA256VM_DIGEST_WIDTH].borrow_mut();
                    digest_cols.from_state.timestamp = F::from_canonical_u32(record.timestamp);
                    digest_cols.from_state.pc = F::from_canonical_u32(record.from_pc);
                    digest_cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                    digest_cols.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
                    digest_cols.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);
                    digest_cols.dst_ptr = record.dst_ptr.to_le_bytes().map(F::from_canonical_u8);
                    digest_cols.src_ptr = record.src_ptr.to_le_bytes().map(F::from_canonical_u8);
                    digest_cols.len_data = record.len.to_le_bytes().map(F::from_canonical_u8);
                    if is_last_block {
                        digest_cols
                            .register_reads_aux
                            .iter_mut()
                            .zip(record.register_reads_aux.iter())
                            .enumerate()
                            .for_each(|(idx, (cols_read, record_read))| {
                                mem_helper.fill(
                                    record_read.prev_timestamp,
                                    record.timestamp + idx as u32,
                                    cols_read.as_mut(),
                                );
                            });
                        digest_cols
                            .writes_aux
                            .set_prev_data(record.write_aux.prev_data.map(F::from_canonical_u8));
                        // In the last block we do `SHA256_NUM_READ_ROWS` reads and then write the
                        // result thus the timestamp of the write is
                        // `block_start_timestamp + SHA256_NUM_READ_ROWS`
                        mem_helper.fill(
                            record.write_aux.prev_timestamp,
                            block_start_timestamp + SHA256_NUM_READ_ROWS as u32,
                            digest_cols.writes_aux.as_mut(),
                        );
                        // Need to range check the destination and source pointers
                        let msl_rshift: u32 =
                            ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS) as u32;
                        let msl_lshift: u32 = (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS
                            - self.pointer_max_bits)
                            as u32;
                        self.bitwise_lookup_chip.request_range(
                            (record.dst_ptr >> msl_rshift) << msl_lshift,
                            (record.src_ptr >> msl_rshift) << msl_lshift,
                        );
                    } else {
                        // Filling in zeros to make sure the accidental garbage data doesn't
                        // overflow the prime
                        digest_cols.register_reads_aux.iter_mut().for_each(|aux| {
                            mem_helper.fill_zero(aux.as_mut());
                        });
                        digest_cols
                            .writes_aux
                            .set_prev_data([F::ZERO; SHA256_WRITE_SIZE]);
                        mem_helper.fill_zero(digest_cols.writes_aux.as_mut());
                    }
                    digest_cols.inner.flags.is_last_block = F::from_bool(is_last_block);
                    digest_cols.inner.flags.is_digest_row = F::from_bool(true);
                } else {
                    // This is a round row
                    let round_cols: &mut Sha256VmRoundCols<F> =
                        row_slice[..SHA256VM_ROUND_WIDTH].borrow_mut();
                    // Take care of the first 4 round rows (aka read rows)
                    if row_idx < SHA256_NUM_READ_ROWS {
                        round_cols
                            .inner
                            .message_schedule
                            .carry_or_buffer
                            .as_flattened_mut()
                            .iter_mut()
                            .zip(
                                input[row_idx * SHA256_READ_SIZE..(row_idx + 1) * SHA256_READ_SIZE]
                                    .iter(),
                            )
                            .for_each(|(cell, data)| {
                                *cell = F::from_canonical_u8(*data);
                            });
                        mem_helper.fill(
                            read_aux_records[row_idx].prev_timestamp,
                            block_start_timestamp + row_idx as u32,
                            round_cols.read_aux.as_mut(),
                        );
                    } else {
                        mem_helper.fill_zero(round_cols.read_aux.as_mut());
                    }
                }
                // Fill in the control cols, doesn't matter if it is a round or digest row
                let control_cols: &mut Sha256VmControlCols<F> =
                    row_slice[..SHA256VM_CONTROL_WIDTH].borrow_mut();
                control_cols.len = F::from_canonical_u32(record.len);
                // Only the first `SHA256_NUM_READ_ROWS` rows increment the timestamp and read ptr
                control_cols.cur_timestamp = F::from_canonical_u32(
                    block_start_timestamp + min(row_idx, SHA256_NUM_READ_ROWS) as u32,
                );
                control_cols.read_ptr = F::from_canonical_u32(
                    block_start_read_ptr
                        + (SHA256_READ_SIZE * min(row_idx, SHA256_NUM_READ_ROWS)) as u32,
                );

                // Fill in the padding flags
                if row_idx < SHA256_NUM_READ_ROWS {
                    #[allow(clippy::comparison_chain)]
                    if (row_idx as i32) < first_padding_row {
                        control_cols.pad_flags = get_flag_pt_array(
                            &self.padding_encoder,
                            PaddingFlags::NotPadding as usize,
                        )
                        .map(F::from_canonical_u32);
                    } else if row_idx as i32 == first_padding_row {
                        let len = message_left - row_idx * SHA256_READ_SIZE;
                        control_cols.pad_flags = get_flag_pt_array(
                            &self.padding_encoder,
                            if row_idx == 3 && is_last_block {
                                PaddingFlags::FirstPadding0_LastRow
                            } else {
                                PaddingFlags::FirstPadding0
                            } as usize
                                + len,
                        )
                        .map(F::from_canonical_u32);
                    } else {
                        control_cols.pad_flags = get_flag_pt_array(
                            &self.padding_encoder,
                            if row_idx == 3 && is_last_block {
                                PaddingFlags::EntirePaddingLastRow
                            } else {
                                PaddingFlags::EntirePadding
                            } as usize,
                        )
                        .map(F::from_canonical_u32);
                    }
                } else {
                    control_cols.pad_flags = get_flag_pt_array(
                        &self.padding_encoder,
                        PaddingFlags::NotConsidered as usize,
                    )
                    .map(F::from_canonical_u32);
                }
                if is_last_block && row_idx == SHA256_ROWS_PER_BLOCK - 1 {
                    // If last digest row, then we set padding_occurred = 0
                    control_cols.padding_occurred = F::ZERO;
                } else {
                    control_cols.padding_occurred =
                        F::from_bool((row_idx as i32) >= first_padding_row);
                }
            });

        // Fill in the inner trace when the `buffer_or_carry` is filled in
        self.inner.generate_block_trace::<F>(
            block_slice,
            SHA256VM_WIDTH,
            SHA256VM_CONTROL_WIDTH,
            &padded_input,
            self.bitwise_lookup_chip.as_ref(),
            &prev_hash,
            is_last_block,
            global_block_idx as u32 + 1, // global block index is 1-indexed
            local_block_idx as u32,
        );
    }
}
