use std::{
    borrow::{Borrow, BorrowMut},
    cmp::min,
    iter,
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, CustomBorrow, MultiRowLayout, MultiRowMetadata, RecordArena, Result,
        SizedRecord, TraceFiller, TraceStep, VmStateMut,
    },
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
use openvm_sha2_air::{
    be_limbs_into_word, get_flag_pt_array, Sha256Config, Sha2StepHelper, Sha384Config, Sha512Config,
};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};

use super::{
    Sha2BlockHasherDigestColsRefMut, Sha2BlockHasherRoundColsRefMut, Sha2ChipConfig, Sha2Variant,
    Sha2VmStep,
};
use crate::{
    get_sha2_num_blocks, sha2_chip::PaddingFlags, sha2_solve, Sha2BlockHasherControlColsRefMut,
    MAX_SHA_NUM_WRITES, SHA_MAX_MESSAGE_LEN, SHA_REGISTER_READS, SHA_WRITE_SIZE,
};

#[derive(Clone, Copy)]
pub struct Sha2VmMetadata<C: Sha2ChipConfig> {
    pub num_blocks: u32,
    _phantom: PhantomData<C>,
}

impl<C: Sha2ChipConfig> MultiRowMetadata for Sha2VmMetadata<C> {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_blocks as usize * C::ROWS_PER_BLOCK
    }
}

pub(crate) type Sha2VmRecordLayout<C> = MultiRowLayout<Sha2VmMetadata<C>>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Sha2VmRecordHeader {
    pub from_pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub dst_ptr: u32,
    pub src_ptr: u32,
    pub len: u32,

    pub register_reads_aux: [MemoryReadAuxRecord; SHA_REGISTER_READS],
    // Note: MAX_SHA_NUM_WRITES = 2 because SHA-256 uses 1 write, while SHA-512 and SHA-384 use 2
    // writes. We just use the same array for all variants to simplify record storage.
    pub writes_aux: [MemoryWriteBytesAuxRecord<SHA_WRITE_SIZE>; MAX_SHA_NUM_WRITES],
}

pub struct Sha2VmRecordMut<'a> {
    pub inner: &'a mut Sha2VmRecordHeader,
    // Having a continuous slice of the input is useful for fast hashing in `execute`
    pub input: &'a mut [u8],
    pub read_aux: &'a mut [MemoryReadAuxRecord],
}

/// Custom borrowing that splits the buffer into a fixed `Sha2VmRecord` header
/// followed by a slice of `u8`'s of length `C::BLOCK_CELLS * num_blocks` where `num_blocks` is
/// provided at runtime, followed by a slice of `MemoryReadAuxRecord`'s of length
/// `C::NUM_READ_ROWS * num_blocks`. Uses `align_to_mut()` to make sure the slice is properly
/// aligned to `MemoryReadAuxRecord`. Has debug assertions that check the size and alignment of the
/// slices.
impl<'a, C: Sha2ChipConfig> CustomBorrow<'a, Sha2VmRecordMut<'a>, Sha2VmRecordLayout<C>>
    for [u8]
{
    fn custom_borrow(&'a mut self, layout: Sha2VmRecordLayout<C>) -> Sha2VmRecordMut<'a> {
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Sha2VmRecordHeader>()) };
        let header: &mut Sha2VmRecordHeader = header_buf.borrow_mut();

        // Using `split_at_mut_unchecked` for perf reasons
        // input is a slice of `u8`'s of length `C::BLOCK_CELLS * num_blocks`, so the alignment
        // is always satisfied
        let (input, rest) = unsafe {
            rest.split_at_mut_unchecked((layout.metadata.num_blocks as usize) * C::BLOCK_CELLS)
        };

        // Using `align_to_mut` to make sure the returned slice is properly aligned to
        // `MemoryReadAuxRecord` Additionally, Rust's subslice operation (a few lines below)
        // will verify that the buffer has enough capacity
        let (_, read_aux_buf, _) = unsafe { rest.align_to_mut::<MemoryReadAuxRecord>() };
        Sha2VmRecordMut {
            inner: header,
            input,
            read_aux: &mut read_aux_buf[..(layout.metadata.num_blocks as usize) * C::NUM_READ_ROWS],
        }
    }

    unsafe fn extract_layout(&self) -> Sha2VmRecordLayout<C> {
        let header: &Sha2VmRecordHeader = self.borrow();

        Sha2VmRecordLayout {
            metadata: Sha2VmMetadata {
                num_blocks: get_sha2_num_blocks::<C>(header.len),
                _phantom: PhantomData::<C>,
            },
        }
    }
}

impl<C: Sha2ChipConfig> SizedRecord<Sha2VmRecordLayout<C>> for Sha2VmRecordMut<'_> {
    fn size(layout: &Sha2VmRecordLayout<C>) -> usize {
        let mut total_len = size_of::<Sha2VmRecordHeader>();
        total_len += layout.metadata.num_blocks as usize * C::BLOCK_CELLS;
        // Align the pointer to the alignment of `MemoryReadAuxRecord`
        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += layout.metadata.num_blocks as usize
            * C::NUM_READ_ROWS
            * size_of::<MemoryReadAuxRecord>();
        total_len
    }

    fn alignment(_layout: &Sha2VmRecordLayout<C>) -> usize {
        align_of::<Sha2VmRecordHeader>()
    }
}

impl<F: PrimeField32, CTX, C: Sha2ChipConfig> TraceStep<F, CTX> for Sha2VmStep<C> {
    type RecordLayout = Sha2VmRecordLayout<C>;
    type RecordMut<'a> = Sha2VmRecordMut<'a>;

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", C::OPCODE)
    }

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(*opcode, C::OPCODE.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Reading the length first to allocate a record of correct size
        let len = read_rv32_register(state.memory.data(), c.as_canonical_u32());

        let num_blocks = get_sha2_num_blocks::<C>(len);
        let record = arena.alloc(MultiRowLayout {
            metadata: Sha2VmMetadata {
                num_blocks,
                _phantom: PhantomData::<C>,
            },
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
            record.inner.src_ptr as usize + num_blocks as usize * C::BLOCK_CELLS
                <= (1 << self.pointer_max_bits)
        );
        debug_assert!(
            record.inner.dst_ptr as usize + C::WRITE_SIZE <= (1 << self.pointer_max_bits)
        );
        // We don't support messages longer than 2^29 bytes
        debug_assert!(record.inner.len < SHA_MAX_MESSAGE_LEN as u32);

        for block_idx in 0..num_blocks as usize {
            // Reads happen on the first 4 rows of each block
            for row in 0..C::NUM_READ_ROWS {
                let read_idx = block_idx * C::NUM_READ_ROWS + row;
                match C::VARIANT {
                    Sha2Variant::Sha256 => {
                        let row_input: [u8; Sha256Config::READ_SIZE] = tracing_read(
                            state.memory,
                            RV32_MEMORY_AS,
                            record.inner.src_ptr + (read_idx * C::READ_SIZE) as u32,
                            &mut record.read_aux[read_idx].prev_timestamp,
                        );
                        record.input[read_idx * C::READ_SIZE..(read_idx + 1) * C::READ_SIZE]
                            .copy_from_slice(&row_input);
                    }
                    Sha2Variant::Sha512 => {
                        let row_input: [u8; Sha512Config::READ_SIZE] = tracing_read(
                            state.memory,
                            RV32_MEMORY_AS,
                            record.inner.src_ptr + (read_idx * C::READ_SIZE) as u32,
                            &mut record.read_aux[read_idx].prev_timestamp,
                        );
                        record.input[read_idx * C::READ_SIZE..(read_idx + 1) * C::READ_SIZE]
                            .copy_from_slice(&row_input);
                    }
                    Sha2Variant::Sha384 => {
                        let row_input: [u8; Sha384Config::READ_SIZE] = tracing_read(
                            state.memory,
                            RV32_MEMORY_AS,
                            record.inner.src_ptr + (read_idx * C::READ_SIZE) as u32,
                            &mut record.read_aux[read_idx].prev_timestamp,
                        );
                        record.input[read_idx * C::READ_SIZE..(read_idx + 1) * C::READ_SIZE]
                            .copy_from_slice(&row_input);
                    }
                }
            }
        }

        let mut output = sha2_solve::<C>(&record.input[..len as usize]);
        match C::VARIANT {
            Sha2Variant::Sha256 => {
                tracing_write::<F, { Sha256Config::WRITE_SIZE }>(
                    state.memory,
                    RV32_MEMORY_AS,
                    record.inner.dst_ptr,
                    output.try_into().unwrap(),
                    &mut record.inner.writes_aux[0].prev_timestamp,
                    &mut record.inner.writes_aux[0].prev_data,
                );
            }
            Sha2Variant::Sha512 => {
                debug_assert!(output.len() % Sha512Config::WRITE_SIZE == 0);
                output
                    .chunks_exact(Sha512Config::WRITE_SIZE)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        tracing_write::<F, { Sha512Config::WRITE_SIZE }>(
                            state.memory,
                            RV32_MEMORY_AS,
                            record.inner.dst_ptr + (i * Sha512Config::WRITE_SIZE) as u32,
                            chunk.try_into().unwrap(),
                            &mut record.inner.writes_aux[i].prev_timestamp,
                            &mut record.inner.writes_aux[i].prev_data,
                        );
                    });
            }
            Sha2Variant::Sha384 => {
                // output is a truncated 48-byte digest, so we will append 16 bytes of zeros
                output.extend(iter::repeat(0).take(16));
                debug_assert!(output.len() % Sha384Config::WRITE_SIZE == 0);
                output
                    .chunks_exact(Sha384Config::WRITE_SIZE)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        tracing_write::<F, { Sha384Config::WRITE_SIZE }>(
                            state.memory,
                            RV32_MEMORY_AS,
                            record.inner.dst_ptr + (i * Sha384Config::WRITE_SIZE) as u32,
                            chunk.try_into().unwrap(),
                            &mut record.inner.writes_aux[i].prev_timestamp,
                            &mut record.inner.writes_aux[i].prev_data,
                        );
                    });
            }
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32, CTX, C: Sha2ChipConfig> TraceFiller<F, CTX> for Sha2VmStep<C> {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let mut chunks = Vec::with_capacity(trace_matrix.height() / C::ROWS_PER_BLOCK);
        let mut sizes = Vec::with_capacity(trace_matrix.height() / C::ROWS_PER_BLOCK);
        let mut trace = &mut trace_matrix.values[..];
        let mut num_blocks_so_far = 0;

        // First pass over the trace to get the number of blocks for each instruction
        // and divide the matrix into chunks of needed sizes
        loop {
            if num_blocks_so_far * C::ROWS_PER_BLOCK >= rows_used {
                // Push all the padding rows as a single chunk and break
                chunks.push(trace);
                sizes.push((0, num_blocks_so_far));
                break;
            } else {
                let record: &Sha2VmRecordHeader = unsafe { get_record_from_slice(&mut trace, ()) };
                let num_blocks = get_sha2_num_blocks::<C>(record.len) as usize;
                let (chunk, rest) =
                    trace.split_at_mut(C::VM_WIDTH * C::ROWS_PER_BLOCK * num_blocks);
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
                if global_block_offset * C::ROWS_PER_BLOCK >= rows_used {
                    // Fill in the invalid rows
                    slice.par_chunks_mut(C::VM_WIDTH).for_each(|row| {
                        // Need to get rid of the accidental garbage data that might overflow the
                        // F's prime field. Unfortunately, there is no good way around this
                        unsafe {
                            std::ptr::write_bytes(
                                row.as_mut_ptr() as *mut u8,
                                0,
                                C::VM_WIDTH * size_of::<F>(),
                            );
                        }
                        let cols = Sha2BlockHasherRoundColsRefMut::<F>::from::<C>(
                            row[..C::VM_ROUND_WIDTH].borrow_mut(),
                        );
                        self.inner.generate_default_row(cols.inner);
                    });
                    return;
                }

                let record: Sha2VmRecordMut = unsafe {
                    get_record_from_slice(
                        slice,
                        Sha2VmRecordLayout {
                            metadata: Sha2VmMetadata {
                                num_blocks: *num_blocks as u32,
                                _phantom: PhantomData::<C>,
                            },
                        },
                    )
                };

                let mut input: Vec<u8> = Vec::with_capacity(C::BLOCK_CELLS * num_blocks);
                input.extend_from_slice(record.input);
                let mut padded_input = input.clone();
                let len = record.inner.len as usize;
                let padded_input_len = padded_input.len();
                padded_input[len] = 1 << (RV32_CELL_BITS - 1);
                padded_input[len + 1..padded_input_len - 4].fill(0);
                padded_input[padded_input_len - 4..]
                    .copy_from_slice(&((len as u32) << 3).to_be_bytes());

                let mut prev_hashes = Vec::with_capacity(*num_blocks);
                prev_hashes.push(C::get_h().to_vec());
                for i in 0..*num_blocks - 1 {
                    prev_hashes.push(Sha2StepHelper::<C>::get_block_hash(
                        &prev_hashes[i],
                        padded_input[i * C::BLOCK_CELLS..(i + 1) * C::BLOCK_CELLS].into(),
                    ));
                }
                // Copy the read aux records and input to another place to safely fill in the trace
                // matrix without overwriting the record
                let mut read_aux_records = Vec::with_capacity(C::NUM_READ_ROWS * num_blocks);
                read_aux_records.extend_from_slice(record.read_aux);
                let vm_record = record.inner.clone();

                slice
                    .par_chunks_exact_mut(C::VM_WIDTH * C::ROWS_PER_BLOCK)
                    .enumerate()
                    .for_each(|(block_idx, block_slice)| {
                        // Need to get rid of the accidental garbage data that might overflow the
                        // F's prime field. Unfortunately, there is no good way around this
                        unsafe {
                            std::ptr::write_bytes(
                                block_slice.as_mut_ptr() as *mut u8,
                                0,
                                C::ROWS_PER_BLOCK * C::VM_WIDTH * size_of::<F>(),
                            );
                        }
                        self.fill_block_trace::<F>(
                            block_slice,
                            &vm_record,
                            &read_aux_records
                                [block_idx * C::NUM_READ_ROWS..(block_idx + 1) * C::NUM_READ_ROWS],
                            &input[block_idx * C::BLOCK_CELLS..(block_idx + 1) * C::BLOCK_CELLS],
                            &padded_input
                                [block_idx * C::BLOCK_CELLS..(block_idx + 1) * C::BLOCK_CELLS],
                            block_idx == *num_blocks - 1,
                            *global_block_offset + block_idx,
                            block_idx,
                            prev_hashes[block_idx].as_slice(),
                            mem_helper,
                        );
                    });
            },
        );

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        trace_matrix.values[C::VM_WIDTH..]
            .par_chunks_mut(C::VM_WIDTH * C::ROWS_PER_BLOCK)
            .take(rows_used / C::ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.inner.generate_missing_cells(
                    chunk,
                    C::VM_WIDTH,
                    C::BLOCK_HASHER_CONTROL_WIDTH,
                );
            });
    }
}

impl<C: Sha2ChipConfig> Sha2VmStep<C> {
    #[allow(clippy::too_many_arguments)]
    fn fill_block_trace<F: PrimeField32>(
        &self,
        block_slice: &mut [F],
        record: &Sha2VmRecordHeader,
        read_aux_records: &[MemoryReadAuxRecord],
        input: &[u8],
        padded_input: &[u8],
        is_last_block: bool,
        global_block_idx: usize,
        local_block_idx: usize,
        prev_hash: &[C::Word],
        mem_helper: &MemoryAuxColsFactory<F>,
    ) {
        debug_assert_eq!(input.len(), C::BLOCK_CELLS);
        debug_assert_eq!(padded_input.len(), C::BLOCK_CELLS);
        debug_assert_eq!(read_aux_records.len(), C::NUM_READ_ROWS);
        debug_assert_eq!(prev_hash.len(), C::HASH_WORDS);

        let padded_input = (0..C::BLOCK_WORDS)
            .map(|i| {
                be_limbs_into_word::<C>(
                    &padded_input[i * C::WORD_U8S..(i + 1) * C::WORD_U8S]
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let block_start_timestamp =
            record.timestamp + (SHA_REGISTER_READS + C::NUM_READ_ROWS * local_block_idx) as u32;

        let read_cells = (C::BLOCK_CELLS * local_block_idx) as u32;
        let block_start_read_ptr = record.src_ptr + read_cells;

        let message_left = if record.len <= read_cells {
            0
        } else {
            (record.len - read_cells) as usize
        };

        // -1 means that padding occurred before the start of the block
        // C::ROWS_PER_BLOCK + 1 means that no padding occurred on this block
        let first_padding_row = if record.len < read_cells {
            -1
        } else if message_left < C::BLOCK_CELLS {
            (message_left / C::READ_SIZE) as i32
        } else {
            (C::ROWS_PER_BLOCK + 1) as i32
        };

        // Fill in the VM columns first because the inner `carry_or_buffer` needs to be filled in
        block_slice
            .par_chunks_exact_mut(C::VM_WIDTH)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                // Handle round rows and digest row separately
                if row_idx == C::ROWS_PER_BLOCK - 1 {
                    // This is a digest row
                    let mut digest_cols = Sha2BlockHasherDigestColsRefMut::<F>::from::<C>(
                        row_slice[..C::VM_DIGEST_WIDTH].borrow_mut(),
                    );
                    digest_cols.from_state.timestamp = F::from_canonical_u32(record.timestamp);
                    digest_cols.from_state.pc = F::from_canonical_u32(record.from_pc);
                    *digest_cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                    *digest_cols.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
                    *digest_cols.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);
                    digest_cols
                        .dst_ptr
                        .iter_mut()
                        .zip(record.dst_ptr.to_le_bytes().map(F::from_canonical_u8))
                        .for_each(|(x, y)| *x = y);
                    digest_cols
                        .src_ptr
                        .iter_mut()
                        .zip(record.src_ptr.to_le_bytes().map(F::from_canonical_u8))
                        .for_each(|(x, y)| *x = y);
                    digest_cols
                        .len_data
                        .iter_mut()
                        .zip(record.len.to_le_bytes().map(F::from_canonical_u8))
                        .for_each(|(x, y)| *x = y);
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
                        for i in 0..C::NUM_WRITES {
                            digest_cols
                                .writes_aux_prev_data
                                .row_mut(i)
                                .iter_mut()
                                .zip(record.writes_aux[i].prev_data.map(F::from_canonical_u8))
                                .for_each(|(x, y)| *x = y);

                            // In the last block we do `C::NUM_READ_ROWS` reads and then write the
                            // result thus the timestamp of the write is
                            // `block_start_timestamp + C::NUM_READ_ROWS`
                            mem_helper.fill(
                                record.writes_aux[i].prev_timestamp,
                                block_start_timestamp + C::NUM_READ_ROWS as u32 + i as u32,
                                &mut digest_cols.writes_aux_base[i],
                            );
                        }
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
                        for i in 0..C::NUM_WRITES {
                            digest_cols.writes_aux_prev_data.row_mut(i).fill(F::ZERO);
                            mem_helper.fill_zero(&mut digest_cols.writes_aux_base[i]);
                        }
                    }
                    *digest_cols.inner.flags.is_last_block = F::from_bool(is_last_block);
                    *digest_cols.inner.flags.is_digest_row = F::from_bool(true);
                } else {
                    // This is a round row
                    let mut round_cols = Sha2BlockHasherRoundColsRefMut::<F>::from::<C>(
                        row_slice[..C::VM_ROUND_WIDTH].borrow_mut(),
                    );
                    // Take care of the first 4 round rows (aka read rows)
                    if row_idx < C::NUM_READ_ROWS {
                        round_cols
                            .inner
                            .message_schedule
                            .carry_or_buffer
                            .iter_mut()
                            .zip(input[row_idx * C::READ_SIZE..(row_idx + 1) * C::READ_SIZE].iter())
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
                let mut control_cols = Sha2BlockHasherControlColsRefMut::<F>::from::<C>(
                    row_slice[..C::BLOCK_HASHER_CONTROL_WIDTH].borrow_mut(),
                );
                *control_cols.len = F::from_canonical_u32(record.len);
                // Only the first `SHA256_NUM_READ_ROWS` rows increment the timestamp and read ptr
                *control_cols.cur_timestamp = F::from_canonical_u32(
                    block_start_timestamp + min(row_idx, C::NUM_READ_ROWS) as u32,
                );
                *control_cols.read_ptr = F::from_canonical_u32(
                    block_start_read_ptr + (C::READ_SIZE * min(row_idx, C::NUM_READ_ROWS)) as u32,
                );

                // Fill in the padding flags
                if row_idx < C::NUM_READ_ROWS {
                    #[allow(clippy::comparison_chain)]
                    if (row_idx as i32) < first_padding_row {
                        control_cols
                            .pad_flags
                            .iter_mut()
                            .zip(
                                get_flag_pt_array(
                                    &self.padding_encoder,
                                    PaddingFlags::NotPadding as usize,
                                )
                                .into_iter()
                                .map(F::from_canonical_u32),
                            )
                            .for_each(|(x, y)| *x = y);
                    } else if row_idx as i32 == first_padding_row {
                        let len = message_left - row_idx * C::READ_SIZE;
                        control_cols
                            .pad_flags
                            .iter_mut()
                            .zip(
                                get_flag_pt_array(
                                    &self.padding_encoder,
                                    if row_idx == 3 && is_last_block {
                                        PaddingFlags::FirstPadding0_LastRow
                                    } else {
                                        PaddingFlags::FirstPadding0
                                    } as usize
                                        + len,
                                )
                                .into_iter()
                                .map(F::from_canonical_u32),
                            )
                            .for_each(|(x, y)| *x = y);
                    } else {
                        control_cols
                            .pad_flags
                            .iter_mut()
                            .zip(
                                get_flag_pt_array(
                                    &self.padding_encoder,
                                    if row_idx == 3 && is_last_block {
                                        PaddingFlags::EntirePaddingLastRow
                                    } else {
                                        PaddingFlags::EntirePadding
                                    } as usize,
                                )
                                .into_iter()
                                .map(F::from_canonical_u32),
                            )
                            .for_each(|(x, y)| *x = y);
                    }
                } else {
                    control_cols
                        .pad_flags
                        .iter_mut()
                        .zip(
                            get_flag_pt_array(
                                &self.padding_encoder,
                                PaddingFlags::NotConsidered as usize,
                            )
                            .into_iter()
                            .map(F::from_canonical_u32),
                        )
                        .for_each(|(x, y)| *x = y);
                }
                if is_last_block && row_idx == C::ROWS_PER_BLOCK - 1 {
                    // If last digest row, then we set padding_occurred = 0
                    *control_cols.padding_occurred = F::ZERO;
                } else {
                    *control_cols.padding_occurred =
                        F::from_bool((row_idx as i32) >= first_padding_row);
                }
            });

        // Fill in the inner trace when the `carry_or_buffer` is filled in
        self.inner.generate_block_trace::<F>(
            block_slice,
            C::VM_WIDTH,
            C::BLOCK_HASHER_CONTROL_WIDTH,
            &padded_input,
            self.bitwise_lookup_chip.clone(),
            prev_hash,
            is_last_block,
            global_block_idx as u32 + 1, // global block index is 1-indexed
            local_block_idx as u32,
        );
    }
}
