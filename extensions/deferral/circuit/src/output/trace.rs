use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
    sync::Arc,
};

use itertools::Itertools;
use openvm_circuit::{
    arch::{
        get_record_from_slice, CustomBorrow, ExecutionError, MultiRowLayout, MultiRowMetadata,
        PreflightExecutor, RecordArena, SizedRecord, TraceFiller, VmField, VmStateMut,
        DEFAULT_BLOCK_SIZE,
    },
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, AlignedBytesBorrow,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{
        RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS,
        RV64_WORD_NUM_LIMBS,
    },
};
use openvm_riscv_circuit::adapters::{
    memory_read, read_rv64_register_as_u32, rv64_bytes_to_u32, tracing_read, tracing_write,
};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    canonicity::CanonicityTraceGen,
    count::DeferralCircuitCountChip,
    output::DeferralOutputCols,
    poseidon2::DeferralPoseidon2Chip,
    utils::{
        f_commit_to_bytes, join_memory_ops, memory_op_chunk, split_output, DIGEST_MEMORY_OPS,
        F_NUM_BYTES, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct DeferralOutputMetadata {
    pub num_rows: usize,
}

impl MultiRowMetadata for DeferralOutputMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_rows
    }
}

pub(crate) type DeferralOutputLayout = MultiRowLayout<DeferralOutputMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct DeferralOutputRecordHeader {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs_ptr: u32,
    pub deferral_idx: u32,
    pub num_rows: u32,

    // Heap pointers and auxiliary records
    pub rd_val: u32,
    pub rs_val: u32,
    pub rd_aux: MemoryReadAuxRecord,
    pub rs_aux: MemoryReadAuxRecord,

    // Output commit and length read auxiliary record
    pub output_commit_and_len_aux: [MemoryReadAuxRecord; OUTPUT_TOTAL_MEMORY_OPS],
}

pub struct DeferralOutputRecordMut<'a> {
    pub header: &'a mut DeferralOutputRecordHeader,
    pub write_bytes: &'a mut [u8],
    pub write_aux: &'a mut [MemoryWriteBytesAuxRecord<DEFAULT_BLOCK_SIZE>],
}

impl<'a> CustomBorrow<'a, DeferralOutputRecordMut<'a>, DeferralOutputLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: DeferralOutputLayout) -> DeferralOutputRecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<DeferralOutputRecordHeader>()) };

        // SAFETY:
        // - The layout guarantees rest has sufficient length for write data
        // - There are DIGEST_SIZE bytes written per row
        let num_write_rows = layout.metadata.num_rows.saturating_sub(1);
        let (write_bytes, rest) =
            unsafe { rest.split_at_mut_unchecked(num_write_rows * DIGEST_SIZE) };

        // SAFETY:
        // - Valid mutable slice from the previous split
        // - Middle slice is properly aligned for MemoryWriteBytesAuxRecord via align_to_mut
        // - Subslice operation [..layout.metadata.num_rows] validates sufficient capacity
        // - Layout calculation ensures space for alignment padding plus required aux records
        let (_, write_aux_buf, _) =
            unsafe { rest.align_to_mut::<MemoryWriteBytesAuxRecord<DEFAULT_BLOCK_SIZE>>() };

        DeferralOutputRecordMut {
            header: header_buf.borrow_mut(),
            write_bytes,
            write_aux: &mut write_aux_buf[..num_write_rows * DIGEST_MEMORY_OPS],
        }
    }

    unsafe fn extract_layout(&self) -> DeferralOutputLayout {
        let record: &DeferralOutputRecordHeader = self.borrow();
        DeferralOutputLayout {
            metadata: DeferralOutputMetadata {
                num_rows: record.num_rows as usize,
            },
        }
    }
}

impl<'a> SizedRecord<DeferralOutputLayout> for DeferralOutputRecordMut<'a> {
    fn size(layout: &DeferralOutputLayout) -> usize {
        let mut total_len = size_of::<DeferralOutputRecordHeader>();
        let num_write_rows = layout.metadata.num_rows.saturating_sub(1);
        total_len += num_write_rows * DIGEST_SIZE;
        total_len =
            total_len.next_multiple_of(align_of::<MemoryWriteBytesAuxRecord<DEFAULT_BLOCK_SIZE>>());
        total_len += num_write_rows
            * DIGEST_MEMORY_OPS
            * size_of::<MemoryWriteBytesAuxRecord<DEFAULT_BLOCK_SIZE>>();
        total_len
    }

    fn alignment(_layout: &DeferralOutputLayout) -> usize {
        align_of::<DeferralOutputRecordHeader>()
    }
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputExecutor;

#[derive(Clone, derive_new::new)]
pub struct DeferralOutputFiller<F: VmField> {
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    address_bits: usize,
}

impl<F, RA> PreflightExecutor<F, RA> for DeferralOutputExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, DeferralOutputLayout, DeferralOutputRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::OUTPUT)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { a, b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        let rd_ptr = a.as_canonical_u32();
        let rs_ptr = b.as_canonical_u32();
        let deferral_idx = c.as_canonical_u32();

        // Do a non-tracing read to get the output_len and compute num_rows
        let read_ptr = read_rv64_register_as_u32(state.memory.data(), rs_ptr);
        let output_key_chunks: [[u8; DEFAULT_BLOCK_SIZE]; OUTPUT_TOTAL_MEMORY_OPS] = from_fn(|i| {
            memory_read(
                state.memory.data(),
                RV64_MEMORY_AS,
                read_ptr + (i * DEFAULT_BLOCK_SIZE) as u32,
            )
        });
        let output_key: [u8; OUTPUT_TOTAL_BYTES] = join_memory_ops(output_key_chunks);
        let (output_commit, output_len) = split_output(output_key);

        let output_len_val = rv64_bytes_to_u32(output_len) as usize;
        let num_rows = output_len_val / DIGEST_SIZE + 1;
        debug_assert!(output_len_val.is_multiple_of(DIGEST_SIZE));

        // We now have the layout and can write the record
        let record = state
            .ctx
            .alloc(DeferralOutputLayout::new(DeferralOutputMetadata {
                num_rows,
            }));

        record.header.from_pc = *state.pc;
        record.header.from_timestamp = state.memory.timestamp();
        record.header.rd_ptr = rd_ptr;
        record.header.rs_ptr = rs_ptr;
        record.header.deferral_idx = deferral_idx;
        record.header.num_rows = num_rows as u32;

        let rd_bytes: [u8; RV64_REGISTER_NUM_LIMBS] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            rd_ptr,
            &mut record.header.rd_aux.prev_timestamp,
        );
        record.header.rd_val = rv64_bytes_to_u32(rd_bytes);

        let rs_bytes: [u8; RV64_REGISTER_NUM_LIMBS] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            rs_ptr,
            &mut record.header.rs_aux.prev_timestamp,
        );
        record.header.rs_val = rv64_bytes_to_u32(rs_bytes);

        let input_ptr = record.header.rs_val;
        let output_ptr = record.header.rd_val;
        for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
            tracing_read::<DEFAULT_BLOCK_SIZE>(
                state.memory,
                RV64_MEMORY_AS,
                input_ptr + (chunk_idx * DEFAULT_BLOCK_SIZE) as u32,
                &mut record.header.output_commit_and_len_aux[chunk_idx].prev_timestamp,
            );
        }

        let output_raw =
            state.streams.deferrals[deferral_idx as usize].get_output(&output_commit.to_vec());
        debug_assert_eq!(output_raw.len(), output_len_val);

        for (row_idx, output_chunk) in output_raw.chunks_exact(DIGEST_SIZE).enumerate() {
            let row_output_ptr = output_ptr + (row_idx * DIGEST_SIZE) as u32;
            for chunk_idx in 0..DIGEST_MEMORY_OPS {
                let aux_idx = row_idx * DIGEST_MEMORY_OPS + chunk_idx;
                tracing_write(
                    state.memory,
                    RV64_MEMORY_AS,
                    row_output_ptr + (chunk_idx * DEFAULT_BLOCK_SIZE) as u32,
                    memory_op_chunk(output_chunk, chunk_idx),
                    &mut record.write_aux[aux_idx].prev_timestamp,
                    &mut record.write_aux[aux_idx].prev_data,
                );
            }
            record.write_bytes[row_idx * DIGEST_SIZE..(row_idx + 1) * DIGEST_SIZE]
                .copy_from_slice(output_chunk);
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F> TraceFiller<F> for DeferralOutputFiller<F>
where
    F: VmField,
{
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace_matrix.width;
        debug_assert_eq!(width, DeferralOutputCols::<u8>::width());

        let mut trace = &mut trace_matrix.values[..width * rows_used];

        while !trace.is_empty() {
            // SAFETY:
            // - Executor writes a valid record to the start of trace
            // - Header is at the start of the record
            let header: &DeferralOutputRecordHeader =
                unsafe { get_record_from_slice(&mut trace, ()) };
            let num_rows = header.num_rows as usize;
            let output_len = (num_rows - 1) * DIGEST_SIZE;
            let (mut section_chunk, rest) = trace.split_at_mut(width * num_rows);

            // Copy write data out first; row filling overwrites the record bytes in-place.
            let (header, write_bytes, write_aux) = {
                // SAFETY:
                // - The section contains exactly one DeferralOutputRecord
                // - Layout is reconstructed from the record header
                let record: DeferralOutputRecordMut = unsafe {
                    get_record_from_slice(
                        &mut section_chunk,
                        DeferralOutputLayout::new(DeferralOutputMetadata { num_rows }),
                    )
                };
                (
                    record.header.clone(),
                    record.write_bytes.to_vec(),
                    record.write_aux.to_vec(),
                )
            };

            // Initial sponge input is [deferral_idx, output_len, 0, ...].
            let mut initial_sponge_input = [F::ZERO; DIGEST_SIZE];
            initial_sponge_input[0] = F::from_u32(header.deferral_idx);
            initial_sponge_input[1] = F::from_usize(output_len);

            let mut current_poseidon2_res = [F::ZERO; DIGEST_SIZE];
            self.count_chip.add_count(header.deferral_idx);

            let output_len_bytes = u32::try_from(output_len)
                .expect("deferral output length should fit a u32")
                .to_le_bytes();
            let output_len_f = output_len_bytes.map(F::from_u8);

            for (row_idx, row) in section_chunk.chunks_exact_mut(width).enumerate() {
                let cols: &mut DeferralOutputCols<F> = row.borrow_mut();

                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);
                cols.is_last = F::from_bool(row_idx + 1 == num_rows);
                cols.section_idx = F::from_usize(row_idx);

                cols.from_state.pc = F::from_u32(header.from_pc);
                cols.from_state.timestamp = F::from_u32(header.from_timestamp);
                cols.rd_ptr = F::from_u32(header.rd_ptr);
                cols.rs_ptr = F::from_u32(header.rs_ptr);
                cols.deferral_idx = F::from_u32(header.deferral_idx);

                cols.rd_val = header.rd_val.to_le_bytes().map(F::from_u8);
                cols.rs_val = header.rs_val.to_le_bytes().map(F::from_u8);

                if row_idx == 0 {
                    debug_assert!(RV64_CELL_BITS * RV64_WORD_NUM_LIMBS >= self.address_bits);
                    let limb_shift_bits = RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.address_bits;

                    self.bitwise_lookup_chip.request_range(
                        (header.rd_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32)
                            << limb_shift_bits,
                        (header.rs_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32)
                            << limb_shift_bits,
                    );
                    self.bitwise_lookup_chip.request_range(
                        (output_len_bytes[F_NUM_BYTES - 1] as u32) << limb_shift_bits,
                        0,
                    );

                    mem_helper.fill(
                        header.rd_aux.prev_timestamp,
                        header.from_timestamp,
                        cols.rd_aux.as_mut(),
                    );
                    mem_helper.fill(
                        header.rs_aux.prev_timestamp,
                        header.from_timestamp + 1,
                        cols.rs_aux.as_mut(),
                    );
                    for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
                        mem_helper.fill(
                            header.output_commit_and_len_aux[chunk_idx].prev_timestamp,
                            header.from_timestamp + 2 + chunk_idx as u32,
                            cols.output_commit_and_len_aux[chunk_idx].as_mut(),
                        );
                    }
                } else {
                    mem_helper.fill_zero(cols.rd_aux.as_mut());
                    mem_helper.fill_zero(cols.rs_aux.as_mut());
                    for chunk_aux in &mut cols.output_commit_and_len_aux {
                        mem_helper.fill_zero(chunk_aux.as_mut());
                    }
                }

                cols.output_len = output_len_f;
                if row_idx == 0 {
                    cols.sponge_inputs = initial_sponge_input;
                    current_poseidon2_res = self.poseidon2_chip.perm_and_record(
                        &cols.sponge_inputs,
                        &[F::ZERO; DIGEST_SIZE],
                        row_idx + 1 == num_rows,
                    );
                    for chunk_aux in &mut cols.write_bytes_aux {
                        mem_helper.fill_zero(chunk_aux.as_mut());
                    }
                } else {
                    let output_chunk =
                        &write_bytes[(row_idx - 1) * DIGEST_SIZE..row_idx * DIGEST_SIZE];
                    for bytes in output_chunk.chunks_exact(2) {
                        self.bitwise_lookup_chip
                            .request_range(bytes[0] as u32, bytes[1] as u32);
                    }
                    cols.sponge_inputs = from_fn(|i| F::from_u8(output_chunk[i]));
                    current_poseidon2_res = self.poseidon2_chip.perm_and_record(
                        &cols.sponge_inputs,
                        &current_poseidon2_res,
                        row_idx + 1 == num_rows,
                    );
                    for chunk_idx in 0..DIGEST_MEMORY_OPS {
                        let aux_idx = (row_idx - 1) * DIGEST_MEMORY_OPS + chunk_idx;
                        cols.write_bytes_aux[chunk_idx]
                            .set_prev_data(write_aux[aux_idx].prev_data.map(F::from_u8));
                        mem_helper.fill(
                            write_aux[aux_idx].prev_timestamp,
                            header.from_timestamp
                                + 2
                                + OUTPUT_TOTAL_MEMORY_OPS as u32
                                + aux_idx as u32,
                            cols.write_bytes_aux[chunk_idx].as_mut(),
                        );
                    }
                }
                cols.poseidon2_res = current_poseidon2_res;
            }

            let output_commit = f_commit_to_bytes(&current_poseidon2_res).map(F::from_u8);
            for row in section_chunk.chunks_exact_mut(width) {
                let cols: &mut DeferralOutputCols<F> = row.borrow_mut();
                cols.output_commit = output_commit;
            }
            let cols: &mut DeferralOutputCols<F> = section_chunk[..width].borrow_mut();
            let output_commit_rcs = output_commit
                .chunks_exact(F_NUM_BYTES)
                .zip(cols.output_commit_lt_aux.iter_mut())
                .map(|(bytes, aux)| {
                    let x_le = from_fn(|i| bytes[i]);
                    CanonicityTraceGen::generate_subrow(&x_le, aux)
                })
                .collect_vec();
            for rc_pair in output_commit_rcs.chunks_exact(2) {
                self.bitwise_lookup_chip
                    .request_range(rc_pair[0], rc_pair[1]);
            }

            trace = rest;
        }

        trace_matrix.values[width * rows_used..].fill(F::ZERO);
    }
}
