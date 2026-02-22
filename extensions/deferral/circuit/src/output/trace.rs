use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, hasher::HasherChip, CustomBorrow, ExecutionError, MultiRowLayout,
        MultiRowMetadata, PreflightExecutor, RecordArena, SizedRecord, TraceFiller, VmField,
        VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::adapters::{
    memory_read, read_rv32_register, tracing_read, tracing_write,
};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    count::DeferralCircuitCountChip,
    output::DeferralOutputCols,
    poseidon2::DeferralPoseidon2Chip,
    utils::{
        f_commit_to_bytes, join_memory_ops, memory_op_chunk, split_output, DIGEST_MEMORY_OPS,
        MEMORY_OP_SIZE, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
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
    pub rd_val: [u8; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [u8; RV32_REGISTER_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxRecord,
    pub rs_aux: MemoryReadAuxRecord,

    // Output commit and length read auxiliary record
    pub output_commit_and_len_aux: [MemoryReadAuxRecord; OUTPUT_TOTAL_MEMORY_OPS],
}

pub struct DeferralOutputRecordMut<'a> {
    pub header: &'a mut DeferralOutputRecordHeader,
    pub write_bytes: &'a mut [u8],
    pub write_aux: &'a mut [MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>],
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
        let (write_bytes, rest) =
            unsafe { rest.split_at_mut_unchecked(layout.metadata.num_rows * DIGEST_SIZE) };

        // SAFETY:
        // - Valid mutable slice from the previous split
        // - Middle slice is properly aligned for MemoryWriteBytesAuxRecord via align_to_mut
        // - Subslice operation [..layout.metadata.num_rows] validates sufficient capacity
        // - Layout calculation ensures space for alignment padding plus required aux records
        let (_, write_aux_buf, _) =
            unsafe { rest.align_to_mut::<MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>>() };

        DeferralOutputRecordMut {
            header: header_buf.borrow_mut(),
            write_bytes,
            write_aux: &mut write_aux_buf[..layout.metadata.num_rows * DIGEST_MEMORY_OPS],
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
        total_len += layout.metadata.num_rows * DIGEST_SIZE;
        total_len =
            total_len.next_multiple_of(align_of::<MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>>());
        total_len += layout.metadata.num_rows
            * DIGEST_MEMORY_OPS
            * size_of::<MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>>();
        total_len
    }

    fn alignment(_layout: &DeferralOutputLayout) -> usize {
        align_of::<DeferralOutputRecordHeader>()
    }
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputExecutor;

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralOutputFiller<F: VmField> {
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
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
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let rd_ptr = a.as_canonical_u32();
        let rs_ptr = b.as_canonical_u32();
        let deferral_idx = c.as_canonical_u32();

        // Do a non-tracing read to get the output_len and compute num_rows
        let read_ptr = read_rv32_register(state.memory.data(), rs_ptr);
        let output_key_chunks: [[u8; MEMORY_OP_SIZE]; OUTPUT_TOTAL_MEMORY_OPS] = from_fn(|i| {
            memory_read(
                state.memory.data(),
                RV32_MEMORY_AS,
                read_ptr + (i * MEMORY_OP_SIZE) as u32,
            )
        });
        let output_key: [u8; OUTPUT_TOTAL_BYTES] = join_memory_ops(output_key_chunks);
        let (output_commit, output_len) = split_output(output_key);

        let output_len_val = u32::from_le_bytes(output_len) as usize;
        let num_rows = output_len_val / DIGEST_SIZE;
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

        record.header.rd_val = tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            rd_ptr,
            &mut record.header.rd_aux.prev_timestamp,
        );
        record.header.rs_val = tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            rs_ptr,
            &mut record.header.rs_aux.prev_timestamp,
        );

        let input_ptr = u32::from_le_bytes(record.header.rs_val);
        let output_ptr = u32::from_le_bytes(record.header.rd_val);
        for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
            tracing_read::<MEMORY_OP_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                input_ptr + (chunk_idx * MEMORY_OP_SIZE) as u32,
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
                    RV32_MEMORY_AS,
                    row_output_ptr + (chunk_idx * MEMORY_OP_SIZE) as u32,
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

            // Starting commit state should be [deferral_idx, 0, ..., 0]
            let mut current_commit_state = [F::ZERO; DIGEST_SIZE];
            current_commit_state[0] = F::from_u32(header.deferral_idx);
            self.count_chip.add_count(header.deferral_idx);

            let output_len_bytes = ((num_rows * DIGEST_SIZE) as u32)
                .to_le_bytes()
                .map(F::from_u8);

            for (row_idx, row) in section_chunk.chunks_exact_mut(width).enumerate() {
                let cols: &mut DeferralOutputCols<F> = row.borrow_mut();

                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);
                cols.section_idx = F::from_usize(row_idx);

                cols.from_state.pc = F::from_u32(header.from_pc);
                cols.from_state.timestamp = F::from_u32(header.from_timestamp);
                cols.rd_ptr = F::from_u32(header.rd_ptr);
                cols.rs_ptr = F::from_u32(header.rs_ptr);
                cols.deferral_idx = F::from_u32(header.deferral_idx);

                cols.rd_val = header.rd_val.map(F::from_u8);
                cols.rs_val = header.rs_val.map(F::from_u8);

                if row_idx == 0 {
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

                cols.output_len.copy_from_slice(&output_len_bytes);
                cols.write_bytes = from_fn(|i| F::from_u8(write_bytes[row_idx * DIGEST_SIZE + i]));

                current_commit_state = self
                    .poseidon2_chip
                    .compress_and_record(&current_commit_state, &cols.write_bytes);
                cols.current_commit_state = current_commit_state;

                for chunk_idx in 0..DIGEST_MEMORY_OPS {
                    let aux_idx = row_idx * DIGEST_MEMORY_OPS + chunk_idx;
                    cols.write_bytes_aux[chunk_idx]
                        .set_prev_data(write_aux[aux_idx].prev_data.map(F::from_u8));
                    mem_helper.fill(
                        write_aux[aux_idx].prev_timestamp,
                        header.from_timestamp + 2 + OUTPUT_TOTAL_MEMORY_OPS as u32 + aux_idx as u32,
                        cols.write_bytes_aux[chunk_idx].as_mut(),
                    );
                }
            }

            let output_commit = f_commit_to_bytes(&current_commit_state).map(F::from_u8);
            for row in section_chunk.chunks_exact_mut(width) {
                let cols: &mut DeferralOutputCols<F> = row.borrow_mut();
                cols.output_commit = output_commit;
            }

            trace = rest;
        }

        trace_matrix.values[width * rows_used..].fill(F::ZERO);
    }
}
