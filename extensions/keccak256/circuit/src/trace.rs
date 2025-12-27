use std::{
    array::{self, from_fn},
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
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};
use p3_keccak_air::{
    generate_trace_rows, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS, U64_LIMBS,
};
use tiny_keccak::keccakf;

use super::{
    columns::KeccakVmCols, KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_RATE_BYTES,
    KECCAK_REGISTER_READS, NUM_ABSORB_ROUNDS,
};
use crate::{
    columns::NUM_KECCAK_VM_COLS,
    utils::{keccak256, keccak_f, num_keccak_f},
    KeccakVmExecutor, KeccakVmFiller, KECCAK_DIGEST_BYTES, KECCAK_RATE_U16S, KECCAK_WORD_SIZE,
};

#[derive(Clone, Copy)]
pub struct KeccakVmMetadata {
    pub len: usize,
}

impl MultiRowMetadata for KeccakVmMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        num_keccak_f(self.len) * NUM_ROUNDS
    }
}

pub(crate) type KeccakVmRecordLayout = MultiRowLayout<KeccakVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct KeccakVmRecordHeader {
    pub from_pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub dst: u32,
    pub src: u32,
    pub len: u32,

    pub register_reads_aux: [MemoryReadAuxRecord; KECCAK_REGISTER_READS],
    pub write_aux: [MemoryWriteBytesAuxRecord<KECCAK_WORD_SIZE>; KECCAK_DIGEST_WRITES],
}

pub struct KeccakVmRecordMut<'a> {
    pub inner: &'a mut KeccakVmRecordHeader,
    // Having a continuous slice of the input is useful for fast hashing in `execute`
    pub input: &'a mut [u8],
    pub read_aux: &'a mut [MemoryReadAuxRecord],
}

/// Custom borrowing that splits the buffer into a fixed `KeccakVmRecord` header
/// followed by a slice of `u8`'s of length `num_reads * KECCAK_WORD_SIZE` where `num_reads` is
/// provided at runtime, followed by a slice of `MemoryReadAuxRecord`'s of length `num_reads`.
/// Uses `align_to_mut()` to make sure the slice is properly aligned to `MemoryReadAuxRecord`.
/// Has debug assertions that check the size and alignment of the slices.
impl<'a> CustomBorrow<'a, KeccakVmRecordMut<'a>, KeccakVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: KeccakVmRecordLayout) -> KeccakVmRecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits and
        //   constants are guaranteed <= self.len() by layout precondition
        let (record_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<KeccakVmRecordHeader>()) };

        let num_reads = layout.metadata.len.div_ceil(KECCAK_WORD_SIZE);
        // Note: each read is `KECCAK_WORD_SIZE` bytes
        // SAFETY:
        // - layout guarantees rest has sufficient length for input data
        // - num_reads is calculated from layout.metadata.len
        // - total buffer size was validated to contain header + input + aligned aux records
        let (input, rest) = unsafe { rest.split_at_mut_unchecked(num_reads * KECCAK_WORD_SIZE) };
        let (_, read_aux_buf, _) = unsafe { rest.align_to_mut::<MemoryReadAuxRecord>() };
        KeccakVmRecordMut {
            inner: record_buf.borrow_mut(),
            input,
            read_aux: &mut read_aux_buf[..num_reads],
        }
    }

    unsafe fn extract_layout(&self) -> KeccakVmRecordLayout {
        let header: &KeccakVmRecordHeader = self.borrow();
        KeccakVmRecordLayout {
            metadata: KeccakVmMetadata {
                len: header.len as usize,
            },
        }
    }
}

impl SizedRecord<KeccakVmRecordLayout> for KeccakVmRecordMut<'_> {
    fn size(layout: &KeccakVmRecordLayout) -> usize {
        let num_reads = layout.metadata.len.div_ceil(KECCAK_WORD_SIZE);
        let mut total_len = size_of::<KeccakVmRecordHeader>();
        total_len += num_reads * KECCAK_WORD_SIZE;
        // Align the pointer to the alignment of `MemoryReadAuxRecord`
        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += num_reads * size_of::<MemoryReadAuxRecord>();
        total_len
    }

    fn alignment(_layout: &KeccakVmRecordLayout) -> usize {
        align_of::<KeccakVmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for KeccakVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, KeccakVmRecordLayout, KeccakVmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32KeccakOpcode::KECCAK256)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(opcode, Rv32KeccakOpcode::KECCAK256.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Reading the length first without tracing to allocate a record of correct size
        let len = read_rv32_register(state.memory.data(), c.as_canonical_u32()) as usize;

        let num_reads = len.div_ceil(KECCAK_WORD_SIZE);
        let num_blocks = num_keccak_f(len);
        let record = state
            .ctx
            .alloc(KeccakVmRecordLayout::new(KeccakVmMetadata { len }));

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

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

        debug_assert!(record.inner.src as usize + len <= (1 << self.pointer_max_bits));
        debug_assert!(
            record.inner.dst as usize + KECCAK_DIGEST_BYTES <= (1 << self.pointer_max_bits)
        );
        // We don't support messages longer than 2^[pointer_max_bits] bytes
        debug_assert!(record.inner.len < (1 << self.pointer_max_bits));

        for idx in 0..num_reads {
            if idx % KECCAK_ABSORB_READS == 0 && idx != 0 {
                // Need to increment the timestamp according at the start of each block due to the
                // AIR constraints
                state
                    .memory
                    .increment_timestamp_by(KECCAK_REGISTER_READS as u32);
            }
            let read = tracing_read::<KECCAK_WORD_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.src + (idx * KECCAK_WORD_SIZE) as u32,
                &mut record.read_aux[idx].prev_timestamp,
            );
            record.input[idx * KECCAK_WORD_SIZE..(idx + 1) * KECCAK_WORD_SIZE]
                .copy_from_slice(&read);
        }

        // Due to the AIR constraints, need to set the timestamp to the following:
        state.memory.timestamp = record.inner.timestamp
            + (num_blocks * (KECCAK_ABSORB_READS + KECCAK_REGISTER_READS)) as u32;

        let digest = keccak256(&record.input[..len]);
        for (i, word) in digest.chunks_exact(KECCAK_WORD_SIZE).enumerate() {
            tracing_write::<KECCAK_WORD_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.dst + (i * KECCAK_WORD_SIZE) as u32,
                word.try_into().unwrap(),
                &mut record.inner.write_aux[i].prev_timestamp,
                &mut record.inner.write_aux[i].prev_data,
            );
        }

        // Due to the AIR constraints, the final memory timestamp should be the following:
        state.memory.timestamp = record.inner.timestamp
            + (len + KECCAK_REGISTER_READS + KECCAK_ABSORB_READS + KECCAK_DIGEST_WRITES) as u32;
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for KeccakVmFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let mut chunks = Vec::with_capacity(trace_matrix.height() / NUM_ROUNDS);
        let mut sizes = Vec::with_capacity(trace_matrix.height() / NUM_ROUNDS);
        let mut trace = &mut trace_matrix.values[..];
        let mut num_blocks_so_far = 0;

        // First pass over the trace to get the number of blocks for each instruction
        // and divide the matrix into chunks of needed sizes
        loop {
            if num_blocks_so_far * NUM_ROUNDS >= rows_used {
                // Push all the dummy rows as a single chunk and break
                chunks.push(trace);
                sizes.push((0, 0));
                break;
            } else {
                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - header is the first element of the record
                let record: &KeccakVmRecordHeader =
                    unsafe { get_record_from_slice(&mut trace, ()) };
                let num_blocks = num_keccak_f(record.len as usize);
                let (chunk, rest) =
                    trace.split_at_mut(NUM_KECCAK_VM_COLS * NUM_ROUNDS * num_blocks);
                chunks.push(chunk);
                sizes.push((num_blocks, record.len as usize));
                num_blocks_so_far += num_blocks;
                trace = rest;
            }
        }

        // First, parallelize over instruction chunks, every instruction can have multiple blocks
        // Then, compute some additional values for each block and parallelize over blocks within an
        // instruction Finally, compute some additional values for each row and parallelize
        // over rows within a block
        chunks
            .par_iter_mut()
            .zip(sizes.par_iter())
            .for_each(|(slice, (num_blocks, len))| {
                if *num_blocks == 0 {
                    // Fill in the dummy rows in parallel
                    // Note: a 'block' of dummy rows is generated by `generate_trace_rows` from the
                    // zero state       dummy rows are repeated every
                    // `NUM_ROUNDS` rows
                    let p3_trace: RowMajorMatrix<F> = generate_trace_rows(vec![[0u64; 25]; 1], 0);

                    slice
                        .par_chunks_exact_mut(NUM_KECCAK_VM_COLS)
                        .enumerate()
                        .for_each(|(row_idx, row)| {
                            let idx = row_idx % NUM_ROUNDS;
                            row[..NUM_KECCAK_PERM_COLS].copy_from_slice(
                                &p3_trace.values
                                    [idx * NUM_KECCAK_PERM_COLS..(idx + 1) * NUM_KECCAK_PERM_COLS],
                            );

                            // Need to get rid of the accidental garbage data that might overflow
                            // the F's prime field. Unfortunately, there
                            // is no good way around this
                            // SAFETY:
                            // - row has exactly NUM_KECCAK_VM_COLS elements
                            // - NUM_KECCAK_PERM_COLS offset is less than NUM_KECCAK_VM_COLS by
                            //   design
                            // - We're zeroing the remaining (NUM_KECCAK_VM_COLS -
                            //   NUM_KECCAK_PERM_COLS) elements to clear any garbage data that might
                            //   overflow the field
                            unsafe {
                                std::ptr::write_bytes(
                                    row.as_mut_ptr().add(NUM_KECCAK_PERM_COLS) as *mut u8,
                                    0,
                                    (NUM_KECCAK_VM_COLS - NUM_KECCAK_PERM_COLS) * size_of::<F>(),
                                );
                            }
                            let cols: &mut KeccakVmCols<F> = row.borrow_mut();
                            // The first row of a `dummy` block should have `is_new_start = F::ONE`
                            cols.sponge.is_new_start = F::from_bool(idx == 0);
                            cols.sponge.block_bytes[0] = F::ONE;
                            cols.sponge.block_bytes[KECCAK_RATE_BYTES - 1] =
                                F::from_canonical_u32(0x80);
                            cols.sponge.is_padding_byte = [F::ONE; KECCAK_RATE_BYTES];
                        });
                    return;
                }

                let num_reads = len.div_ceil(KECCAK_WORD_SIZE);
                let read_len = num_reads * KECCAK_WORD_SIZE;

                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - slice contains a valid KeccakVmRecord with the exact layout specified
                // - get_record_from_slice will correctly split the buffer into header, input, and
                //   aux components based on this layout
                let record: KeccakVmRecordMut = unsafe {
                    get_record_from_slice(
                        slice,
                        KeccakVmRecordLayout::new(KeccakVmMetadata { len: *len }),
                    )
                };

                // Copy the read aux records and inner record to another place
                // to safely fill in the trace matrix without overwriting the record
                let mut read_aux_records = Vec::with_capacity(num_reads);
                read_aux_records.extend_from_slice(record.read_aux);
                let vm_record = record.inner.clone();
                let partial_block = if read_len != *len {
                    record.input[read_len - KECCAK_WORD_SIZE + 1..]
                        .try_into()
                        .unwrap()
                } else {
                    [0u8; KECCAK_WORD_SIZE - 1]
                }
                .map(F::from_canonical_u8);
                let mut input = Vec::with_capacity(*num_blocks * KECCAK_RATE_BYTES);
                input.extend_from_slice(&record.input[..*len]);
                // Pad the input according to the Keccak spec
                input.push(0x01);
                input.resize(input.capacity(), 0);
                *input.last_mut().unwrap() += 0x80;

                let mut states = Vec::with_capacity(*num_blocks);
                let mut state = [0u64; 25];

                input
                    .chunks_exact(KECCAK_RATE_BYTES)
                    .enumerate()
                    .for_each(|(idx, chunk)| {
                        // absorb
                        for (bytes, s) in chunk.chunks_exact(8).zip(state.iter_mut()) {
                            // u64 <-> bytes conversion is little-endian
                            for (i, &byte) in bytes.iter().enumerate() {
                                let s_byte = (*s >> (i * 8)) as u8;
                                // Update bitwise lookup (i.e. xor) chip state: order matters!
                                if idx != 0 {
                                    self.bitwise_lookup_chip
                                        .request_xor(byte as u32, s_byte as u32);
                                }
                                *s ^= (byte as u64) << (i * 8);
                            }
                        }
                        states.push(state);
                        keccakf(&mut state);
                    });

                slice
                    .par_chunks_exact_mut(NUM_ROUNDS * NUM_KECCAK_VM_COLS)
                    .enumerate()
                    .for_each(|(block_idx, block_slice)| {
                        // We need to transpose state matrices due to a plonky3 issue: https://github.com/Plonky3/Plonky3/issues/672
                        // Note: the fix for this issue will be a commit after the major Field crate refactor PR https://github.com/Plonky3/Plonky3/pull/640
                        //       which will require a significant refactor to switch to.
                        let state = from_fn(|i| {
                            let x = i / 5;
                            let y = i % 5;
                            states[block_idx][x + 5 * y]
                        });

                        // Note: we can call `generate_trace_rows` for each block separately because
                        // its trace only depends on the current `state`
                        // `generate_trace_rows` will generate additional dummy rows to make the
                        // height into power of 2, but we can safely discard them
                        let p3_trace: RowMajorMatrix<F> = generate_trace_rows(vec![state], 0);
                        let input_offset = block_idx * KECCAK_RATE_BYTES;
                        let start_timestamp = vm_record.timestamp
                            + (block_idx * (KECCAK_REGISTER_READS + KECCAK_ABSORB_READS)) as u32;
                        let rem_len = *len - input_offset;

                        block_slice
                            .par_chunks_exact_mut(NUM_KECCAK_VM_COLS)
                            .enumerate()
                            .zip(p3_trace.values.par_chunks(NUM_KECCAK_PERM_COLS))
                            .for_each(|((row_idx, row), p3_row)| {
                                // Fill the inner columns
                                // Safety: `KeccakPermCols` **must** be the first field in
                                // `KeccakVmCols`
                                row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_row);

                                let cols: &mut KeccakVmCols<F> = row.borrow_mut();
                                // Fill the sponge columns
                                cols.sponge.is_new_start =
                                    F::from_bool(block_idx == 0 && row_idx == 0);
                                if rem_len < KECCAK_RATE_BYTES {
                                    cols.sponge.is_padding_byte[..rem_len].fill(F::ZERO);
                                    cols.sponge.is_padding_byte[rem_len..].fill(F::ONE);
                                } else {
                                    cols.sponge.is_padding_byte = [F::ZERO; KECCAK_RATE_BYTES];
                                }
                                cols.sponge.block_bytes = array::from_fn(|i| {
                                    F::from_canonical_u8(input[input_offset + i])
                                });
                                if row_idx == 0 {
                                    cols.sponge.state_hi = from_fn(|i| {
                                        F::from_canonical_u8(
                                            (states[block_idx][i / U64_LIMBS]
                                                >> ((i % U64_LIMBS) * 16 + 8))
                                                as u8,
                                        )
                                    });
                                } else if row_idx == NUM_ROUNDS - 1 {
                                    let state = keccak_f(states[block_idx]);
                                    cols.sponge.state_hi = from_fn(|i| {
                                        F::from_canonical_u8(
                                            (state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8))
                                                as u8,
                                        )
                                    });
                                    if block_idx == num_blocks - 1 {
                                        cols.inner.export = F::ONE;
                                        for s in state.into_iter().take(NUM_ABSORB_ROUNDS) {
                                            for s_byte in s.to_le_bytes() {
                                                self.bitwise_lookup_chip
                                                    .request_xor(0, s_byte as u32);
                                            }
                                        }
                                    }
                                } else {
                                    cols.sponge.state_hi = [F::ZERO; KECCAK_RATE_U16S];
                                }

                                // Fill the instruction columns
                                cols.instruction.pc = F::from_canonical_u32(vm_record.from_pc);
                                cols.instruction.is_enabled = F::ONE;
                                cols.instruction.is_enabled_first_round =
                                    F::from_bool(row_idx == 0);
                                cols.instruction.start_timestamp =
                                    F::from_canonical_u32(start_timestamp);
                                cols.instruction.dst_ptr = F::from_canonical_u32(vm_record.rd_ptr);
                                cols.instruction.src_ptr = F::from_canonical_u32(vm_record.rs1_ptr);
                                cols.instruction.len_ptr = F::from_canonical_u32(vm_record.rs2_ptr);
                                cols.instruction.dst =
                                    vm_record.dst.to_le_bytes().map(F::from_canonical_u8);

                                let src = vm_record.src + (block_idx * KECCAK_RATE_BYTES) as u32;
                                cols.instruction.src = F::from_canonical_u32(src);
                                cols.instruction.src_limbs.copy_from_slice(
                                    &src.to_le_bytes().map(F::from_canonical_u8)[1..],
                                );
                                cols.instruction.len_limbs.copy_from_slice(
                                    &(rem_len as u32).to_le_bytes().map(F::from_canonical_u8)[1..],
                                );
                                cols.instruction.remaining_len =
                                    F::from_canonical_u32(rem_len as u32);

                                // Fill the register reads
                                if row_idx == 0 && block_idx == 0 {
                                    for ((i, cols), vm_record) in cols
                                        .mem_oc
                                        .register_aux
                                        .iter_mut()
                                        .enumerate()
                                        .zip(vm_record.register_reads_aux.iter())
                                    {
                                        mem_helper.fill(
                                            vm_record.prev_timestamp,
                                            start_timestamp + i as u32,
                                            cols.as_mut(),
                                        );
                                    }

                                    let msl_rshift = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
                                    let msl_lshift = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS
                                        - self.pointer_max_bits;
                                    // Update the bitwise lookup chip
                                    self.bitwise_lookup_chip.request_range(
                                        (vm_record.dst >> msl_rshift) << msl_lshift,
                                        (vm_record.src >> msl_rshift) << msl_lshift,
                                    );
                                    self.bitwise_lookup_chip.request_range(
                                        (vm_record.len >> msl_rshift) << msl_lshift,
                                        (vm_record.len >> msl_rshift) << msl_lshift,
                                    );
                                } else {
                                    cols.mem_oc.register_aux.par_iter_mut().for_each(|aux| {
                                        mem_helper.fill_zero(aux.as_mut());
                                    });
                                }

                                // Fill the absorb reads
                                if row_idx == 0 {
                                    let reads_offs = block_idx * KECCAK_ABSORB_READS;
                                    let num_reads = min(
                                        rem_len.div_ceil(KECCAK_WORD_SIZE),
                                        KECCAK_ABSORB_READS,
                                    );
                                    let start_timestamp =
                                        start_timestamp + KECCAK_REGISTER_READS as u32;
                                    for i in 0..num_reads {
                                        mem_helper.fill(
                                            read_aux_records[i + reads_offs].prev_timestamp,
                                            start_timestamp + i as u32,
                                            cols.mem_oc.absorb_reads[i].as_mut(),
                                        );
                                    }
                                    for i in num_reads..KECCAK_ABSORB_READS {
                                        mem_helper.fill_zero(cols.mem_oc.absorb_reads[i].as_mut());
                                    }
                                } else {
                                    cols.mem_oc.absorb_reads.par_iter_mut().for_each(|aux| {
                                        mem_helper.fill_zero(aux.as_mut());
                                    });
                                }

                                if block_idx == num_blocks - 1 && row_idx == NUM_ROUNDS - 1 {
                                    let timestamp = start_timestamp
                                        + (KECCAK_ABSORB_READS + KECCAK_REGISTER_READS) as u32;
                                    cols.mem_oc
                                        .digest_writes
                                        .par_iter_mut()
                                        .enumerate()
                                        .zip(vm_record.write_aux.par_iter())
                                        .for_each(|((i, cols), vm_record)| {
                                            cols.set_prev_data(
                                                vm_record.prev_data.map(F::from_canonical_u8),
                                            );
                                            mem_helper.fill(
                                                vm_record.prev_timestamp,
                                                timestamp + i as u32,
                                                cols.as_mut(),
                                            );
                                        });
                                } else {
                                    cols.mem_oc.digest_writes.par_iter_mut().for_each(|aux| {
                                        aux.set_prev_data([F::ZERO; KECCAK_WORD_SIZE]);
                                        mem_helper.fill_zero(aux.as_mut());
                                    });
                                }

                                // Set the partial block only for the last block
                                if block_idx == num_blocks - 1 {
                                    cols.mem_oc.partial_block = partial_block;
                                } else {
                                    cols.mem_oc.partial_block = [F::ZERO; KECCAK_WORD_SIZE - 1];
                                }
                            });
                    });
            });
    }
}
