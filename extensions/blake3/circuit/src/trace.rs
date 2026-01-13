use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
};

use openvm_blake3_transpiler::Rv32Blake3Opcode;
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
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};
use p3_blake3_air::{generate_trace_rows, NUM_BLAKE3_COLS as NUM_BLAKE3_COMPRESS_COLS};

use super::{
    columns::Blake3VmCols, BLAKE3_BLOCK_BYTES, BLAKE3_DIGEST_WRITES, BLAKE3_INPUT_READS,
    BLAKE3_REGISTER_READS, BLAKE3_WORD_SIZE,
};
use crate::{
    columns::NUM_BLAKE3_VM_COLS,
    utils::{blake3_compress, blake3_hash_p3_full_blocks, num_blake3_compressions, BLAKE3_IV},
    Blake3VmExecutor, Blake3VmFiller,
};

/// Metadata about a single BLAKE3 hash operation
#[derive(Clone, Copy)]
pub struct Blake3VmMetadata {
    pub len: usize,
}

impl MultiRowMetadata for Blake3VmMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        // One row per compression
        num_blake3_compressions(self.len)
    }
}

pub(crate) type Blake3VmRecordLayout = MultiRowLayout<Blake3VmMetadata>;

/// Fixed-size header for each hash record
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Blake3VmRecordHeader {
    pub from_pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub dst: u32,
    pub src: u32,
    pub len: u32,

    pub register_reads_aux: [MemoryReadAuxRecord; BLAKE3_REGISTER_READS],
    pub write_aux: [MemoryWriteBytesAuxRecord<BLAKE3_WORD_SIZE>; BLAKE3_DIGEST_WRITES],
}

/// Mutable view into a BLAKE3 execution record
pub struct Blake3VmRecordMut<'a> {
    pub inner: &'a mut Blake3VmRecordHeader,
    pub input: &'a mut [u8],
    pub read_aux: &'a mut [MemoryReadAuxRecord],
}

impl<'a> CustomBorrow<'a, Blake3VmRecordMut<'a>, Blake3VmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Blake3VmRecordLayout) -> Blake3VmRecordMut<'a> {
        let (record_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Blake3VmRecordHeader>()) };

        // Allocate full blocks worth of input and aux records
        let num_blocks = num_blake3_compressions(layout.metadata.len);
        let total_input_bytes = num_blocks * BLAKE3_BLOCK_BYTES;
        let total_aux_records = num_blocks * BLAKE3_INPUT_READS;

        let (input, rest) = unsafe { rest.split_at_mut_unchecked(total_input_bytes) };
        let (_, read_aux_buf, _) = unsafe { rest.align_to_mut::<MemoryReadAuxRecord>() };

        Blake3VmRecordMut {
            inner: record_buf.borrow_mut(),
            input,
            read_aux: &mut read_aux_buf[..total_aux_records],
        }
    }

    unsafe fn extract_layout(&self) -> Blake3VmRecordLayout {
        let header: &Blake3VmRecordHeader = self.borrow();
        Blake3VmRecordLayout {
            metadata: Blake3VmMetadata {
                len: header.len as usize,
            },
        }
    }
}

impl SizedRecord<Blake3VmRecordLayout> for Blake3VmRecordMut<'_> {
    fn size(layout: &Blake3VmRecordLayout) -> usize {
        // Allocate full blocks worth of space
        let num_blocks = num_blake3_compressions(layout.metadata.len);
        let total_input_bytes = num_blocks * BLAKE3_BLOCK_BYTES;
        let total_aux_records = num_blocks * BLAKE3_INPUT_READS;

        let mut total_len = size_of::<Blake3VmRecordHeader>();
        total_len += total_input_bytes;
        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += total_aux_records * size_of::<MemoryReadAuxRecord>();
        total_len
    }

    fn alignment(_layout: &Blake3VmRecordLayout) -> usize {
        align_of::<Blake3VmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for Blake3VmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, Blake3VmRecordLayout, Blake3VmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32Blake3Opcode::BLAKE3)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, b, c, .. } = instruction;

        // Read length first to allocate correct record size
        let len = read_rv32_register(state.memory.data(), c.as_canonical_u32()) as usize;
        let num_blocks = num_blake3_compressions(len);

        // Allocate record
        let record = state
            .ctx
            .alloc(Blake3VmRecordLayout::new(Blake3VmMetadata { len }));

        // Fill header
        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

        // Read registers with tracing
        record.inner.dst = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rd_ptr,
            &mut record.inner.register_reads_aux[0].prev_timestamp,
        ));
        record.inner.src = u32::from_le_bytes(tracing_read(
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

        // Read input data with tracing
        // Each block has BLAKE3_INPUT_READS (16) words, and the AIR expects timestamps
        // to be spaced as (REGISTER_READS + INPUT_READS) per block.
        // We need to read inputs block by block with proper timestamp spacing.
        let block_timestamp_delta = BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS;

        for block_idx in 0..num_blocks {
            let block_start_read = block_idx * BLAKE3_INPUT_READS;
            // Set timestamp to what this block expects for input reads
            // Block 0: start_timestamp + REGISTER_READS (registers already read)
            // Block N: start_timestamp + N * block_timestamp_delta + REGISTER_READS
            let expected_input_timestamp = record.inner.timestamp
                + (block_idx * block_timestamp_delta + BLAKE3_REGISTER_READS) as u32;
            state.memory.timestamp = expected_input_timestamp;

            // Read ALL 16 words for the block (including padding)
            // The AIR constrains reads for the full block, so we must read all
            // values from memory, even beyond the message length
            for i in 0..BLAKE3_INPUT_READS {
                let idx = block_start_read + i;
                let read = tracing_read::<BLAKE3_WORD_SIZE>(
                    state.memory,
                    RV32_MEMORY_AS,
                    record.inner.src + (idx * BLAKE3_WORD_SIZE) as u32,
                    &mut record.read_aux[idx].prev_timestamp,
                );
                // Copy ALL read values (including padding) so trace generation
                // uses the same values that memory actually contains
                record.input[idx * BLAKE3_WORD_SIZE..(idx + 1) * BLAKE3_WORD_SIZE]
                    .copy_from_slice(&read);
            }
        }

        // Update timestamp to be after all blocks
        state.memory.timestamp =
            record.inner.timestamp + (num_blocks * block_timestamp_delta) as u32;

        // Compute hash using p3-compatible parameters over full blocks (with actual memory values)
        // The AIR uses actual memory values (including garbage after message), so we must too
        let full_blocks_len = num_blocks * BLAKE3_BLOCK_BYTES;
        let digest = blake3_hash_p3_full_blocks(&record.input[..full_blocks_len]);
        for (i, word) in digest.chunks_exact(BLAKE3_WORD_SIZE).enumerate() {
            tracing_write::<BLAKE3_WORD_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.dst + (i * BLAKE3_WORD_SIZE) as u32,
                word.try_into().unwrap(),
                &mut record.inner.write_aux[i].prev_timestamp,
                &mut record.inner.write_aux[i].prev_data,
            );
        }

        // Final timestamp: must match AIR's timestamp_change formula
        // timestamp_change = len + REGISTER_READS + INPUT_READS + DIGEST_WRITES
        state.memory.timestamp = record.inner.timestamp
            + (len + BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS + BLAKE3_DIGEST_WRITES) as u32;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for Blake3VmFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let mut chunks = Vec::with_capacity(trace_matrix.height());
        let mut sizes = Vec::with_capacity(trace_matrix.height());
        let mut trace = &mut trace_matrix.values[..];
        let mut rows_so_far = 0;

        // First pass: split trace into chunks per instruction
        loop {
            if rows_so_far >= rows_used {
                // Push all dummy rows as a single chunk
                chunks.push(trace);
                sizes.push((0, 0));
                break;
            } else {
                let record: &Blake3VmRecordHeader =
                    unsafe { get_record_from_slice(&mut trace, ()) };
                let num_blocks = num_blake3_compressions(record.len as usize);
                let (chunk, rest) = trace.split_at_mut(NUM_BLAKE3_VM_COLS * num_blocks);
                chunks.push(chunk);
                sizes.push((num_blocks, record.len as usize));
                rows_so_far += num_blocks;
                trace = rest;
            }
        }

        // Parallel processing of chunks
        chunks
            .par_iter_mut()
            .zip(sizes.par_iter())
            .for_each(|(slice, (num_blocks, len))| {
                if *num_blocks == 0 {
                    // Fill dummy rows
                    let dummy_inputs: Vec<[u32; 24]> = vec![[0u32; 24]];
                    let p3_trace: RowMajorMatrix<F> = generate_trace_rows(dummy_inputs, 0);

                    slice
                        .par_chunks_exact_mut(NUM_BLAKE3_VM_COLS)
                        .for_each(|row| {
                            row[..NUM_BLAKE3_COMPRESS_COLS]
                                .copy_from_slice(&p3_trace.values[..NUM_BLAKE3_COMPRESS_COLS]);

                            // Zero out VM-specific columns
                            unsafe {
                                std::ptr::write_bytes(
                                    row.as_mut_ptr().add(NUM_BLAKE3_COMPRESS_COLS) as *mut u8,
                                    0,
                                    (NUM_BLAKE3_VM_COLS - NUM_BLAKE3_COMPRESS_COLS)
                                        * size_of::<F>(),
                                );
                            }
                            let cols: &mut Blake3VmCols<F> = row.borrow_mut();
                            // Dummy rows: is_enabled=0 (from zero out), is_new_start=0
                            // The AIR constraint expects: is_new_start + (1 - is_enabled) = 1
                            // For disabled rows: 0 + (1 - 0) = 1 ✓
                            cols.instruction.is_new_start = F::ZERO;
                            cols.instruction.is_last_block = F::ONE;
                        });
                    return;
                }

                let num_reads = len.div_ceil(BLAKE3_WORD_SIZE);

                let record: Blake3VmRecordMut = unsafe {
                    get_record_from_slice(
                        slice,
                        Blake3VmRecordLayout::new(Blake3VmMetadata { len: *len }),
                    )
                };

                // Copy record data before overwriting
                let read_aux_records: Vec<_> = record.read_aux.iter().cloned().collect();
                let vm_record = record.inner.clone();
                // Use the full input from execution (which contains actual memory values)
                // This includes padding bytes as they exist in memory
                let input: Vec<u8> = record.input[..*num_blocks * BLAKE3_BLOCK_BYTES].to_vec();

                // Pre-compute chaining values for each block by generating p3 traces
                // sequentially and extracting outputs.
                // This is necessary because block N+1's chaining value depends on block N's outputs.
                let mut chaining_values = Vec::with_capacity(*num_blocks);
                // Pre-compute chaining values sequentially (like Keccak's keccakf)
                // This is identical to how keccak256 computes states before trace generation
                let mut cv = BLAKE3_IV;

                for block_idx in 0..*num_blocks {
                    chaining_values.push(cv);

                    // Skip computing next CV for the last block
                    if block_idx >= *num_blocks - 1 {
                        continue;
                    }

                    // Extract the 64-byte block for this iteration
                    let block_start = block_idx * BLAKE3_BLOCK_BYTES;
                    let block: [u8; BLAKE3_BLOCK_BYTES] = input
                        [block_start..block_start + BLAKE3_BLOCK_BYTES]
                        .try_into()
                        .unwrap();

                    // Compute compression to get next CV
                    blake3_compress(
                        &mut cv, &block, 1, // block_len = num_rows in vec = 1
                        0, // counter = enumerate index = 0
                        0, // flags
                    );
                }

                // Fill each row (one per block)
                slice
                    .par_chunks_exact_mut(NUM_BLAKE3_VM_COLS)
                    .enumerate()
                    .for_each(|(block_idx, row)| {
                        let input_offset = block_idx * BLAKE3_BLOCK_BYTES;
                        let rem_len = len.saturating_sub(input_offset);
                        let is_last_block = block_idx == num_blocks - 1;
                        let is_first_block = block_idx == 0;

                        // Prepare compression input for p3-blake3-air
                        let block_start = block_idx * BLAKE3_BLOCK_BYTES;
                        let mut msg_words = [0u32; 16];
                        for (i, chunk) in input[block_start..block_start + BLAKE3_BLOCK_BYTES]
                            .chunks_exact(4)
                            .enumerate()
                        {
                            msg_words[i] = u32::from_le_bytes(chunk.try_into().unwrap());
                        }

                        // Build the [u32; 24] input for p3-blake3-air
                        let cv = chaining_values[block_idx];
                        let mut compression_input = [0u32; 24];
                        compression_input[..16].copy_from_slice(&msg_words);
                        compression_input[16..24].copy_from_slice(&cv);

                        // Generate trace for this compression
                        let p3_trace: RowMajorMatrix<F> =
                            generate_trace_rows(vec![compression_input], 0);
                        row[..NUM_BLAKE3_COMPRESS_COLS]
                            .copy_from_slice(&p3_trace.values[..NUM_BLAKE3_COMPRESS_COLS]);

                        let cols: &mut Blake3VmCols<F> = row.borrow_mut();

                        // Fill instruction columns
                        cols.instruction.pc = F::from_canonical_u32(vm_record.from_pc);
                        cols.instruction.is_enabled = F::ONE;
                        cols.instruction.is_new_start = F::from_bool(is_first_block);
                        cols.instruction.is_last_block = F::from_bool(is_last_block);
                        cols.instruction.is_enabled_first_block = F::from_bool(is_first_block);

                        let start_timestamp = vm_record.timestamp
                            + (block_idx * (BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS)) as u32;
                        cols.instruction.start_timestamp = F::from_canonical_u32(start_timestamp);

                        cols.instruction.dst_ptr = F::from_canonical_u32(vm_record.rd_ptr);
                        cols.instruction.src_ptr = F::from_canonical_u32(vm_record.rs1_ptr);
                        cols.instruction.len_ptr = F::from_canonical_u32(vm_record.rs2_ptr);
                        cols.instruction.dst =
                            vm_record.dst.to_le_bytes().map(F::from_canonical_u8);

                        let src = vm_record.src + (block_idx * BLAKE3_BLOCK_BYTES) as u32;
                        cols.instruction.src = F::from_canonical_u32(src);
                        cols.instruction
                            .src_limbs
                            .copy_from_slice(&src.to_le_bytes().map(F::from_canonical_u8)[1..]);
                        cols.instruction.len_limbs.copy_from_slice(
                            &(rem_len as u32).to_le_bytes().map(F::from_canonical_u8)[1..],
                        );
                        cols.instruction.remaining_len = F::from_canonical_u32(rem_len as u32);

                        // Fill register reads (only on first block)
                        if is_first_block {
                            for (i, (aux, record)) in cols
                                .mem_oc
                                .register_aux
                                .iter_mut()
                                .zip(vm_record.register_reads_aux.iter())
                                .enumerate()
                            {
                                mem_helper.fill(
                                    record.prev_timestamp,
                                    start_timestamp + i as u32,
                                    aux.as_mut(),
                                );
                            }

                            // Range check MSB limbs
                            let msl_rshift = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
                            let msl_lshift =
                                RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
                            self.bitwise_lookup_chip.request_range(
                                (vm_record.dst >> msl_rshift) << msl_lshift,
                                (vm_record.src >> msl_rshift) << msl_lshift,
                            );
                            self.bitwise_lookup_chip.request_range(
                                (vm_record.len >> msl_rshift) << msl_lshift,
                                (vm_record.len >> msl_rshift) << msl_lshift,
                            );
                        } else {
                            cols.mem_oc.register_aux.iter_mut().for_each(|aux| {
                                mem_helper.fill_zero(aux.as_mut());
                            });
                        }

                        // Fill input reads - ALL 16 words per block
                        let reads_offset = block_idx * BLAKE3_INPUT_READS;
                        let input_timestamp = start_timestamp + BLAKE3_REGISTER_READS as u32;

                        // Fill all 16 input read aux columns
                        for i in 0..BLAKE3_INPUT_READS {
                            let record_idx = reads_offset + i;
                            if record_idx < read_aux_records.len() {
                                mem_helper.fill(
                                    read_aux_records[record_idx].prev_timestamp,
                                    input_timestamp + i as u32,
                                    cols.mem_oc.input_reads[i].as_mut(),
                                );
                            } else {
                                mem_helper.fill_zero(cols.mem_oc.input_reads[i].as_mut());
                            }
                        }

                        // Fill digest writes (only on last block)
                        if is_last_block {
                            let write_timestamp = start_timestamp
                                + (BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS) as u32;
                            for (i, (aux, record)) in cols
                                .mem_oc
                                .digest_writes
                                .iter_mut()
                                .zip(vm_record.write_aux.iter())
                                .enumerate()
                            {
                                aux.set_prev_data(record.prev_data.map(F::from_canonical_u8));
                                mem_helper.fill(
                                    record.prev_timestamp,
                                    write_timestamp + i as u32,
                                    aux.as_mut(),
                                );
                            }
                        } else {
                            cols.mem_oc.digest_writes.iter_mut().for_each(|aux| {
                                aux.set_prev_data([F::ZERO; BLAKE3_WORD_SIZE]);
                                mem_helper.fill_zero(aux.as_mut());
                            });
                        }

                        // Fill partial block (for handling unaligned last read)
                        if is_last_block && *len % BLAKE3_WORD_SIZE != 0 {
                            let read_len = num_reads * BLAKE3_WORD_SIZE;
                            cols.mem_oc.partial_block = from_fn(|i| {
                                if i + 1 < BLAKE3_WORD_SIZE
                                    && read_len - BLAKE3_WORD_SIZE + 1 + i < *len
                                {
                                    F::from_canonical_u8(input[read_len - BLAKE3_WORD_SIZE + 1 + i])
                                } else {
                                    F::ZERO
                                }
                            });
                        } else {
                            cols.mem_oc.partial_block = [F::ZERO; BLAKE3_WORD_SIZE - 1];
                        }
                    });
            });
    }
}
