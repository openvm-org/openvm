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
        Postflight, PostflightError, PreflightExecutor, RecordArena, SizedRecord, TraceFiller,
        U16Access, VmField, VmStateMut, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{pack_u8_block_bytes, MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, AlignedBytesBorrow,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{
        RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS,
        RV64_WORD_NUM_LIMBS,
    },
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::{
    memory_read, read_rv64_register_as_u32, rv64_bytes_to_u32, rv64_u16_block_to_bytes,
    tracing_read, tracing_write,
};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    canonicity::CanonicityTraceGen,
    count::DeferralCircuitCountChip,
    output::{checked_deferral_index, DeferralOutputChip, DeferralOutputCols},
    poseidon2::DeferralPoseidon2Chip,
    utils::{
        byte_memory_op_chunk, checked_pointer_offset, checked_u16_pointer, f_commit_to_bytes,
        join_byte_memory_ops, logged_u32_pointer, require_block_alignment, split_output,
        DIGEST_BYTE_MEMORY_OPS, F_NUM_BYTES, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
};

struct DeferralOutputReplay {
    from_pc: u32,
    from_timestamp: u32,
    rd_ptr: u32,
    rs_ptr: u32,
    deferral_idx: u32,
    rd_val: u32,
    rs_val: u32,
    rd: U16Access,
    rs: U16Access,
    output_commit: [u8; crate::utils::COMMIT_NUM_BYTES],
    output_len: u32,
    output_key_accesses: Vec<U16Access>,
    output_chunks: Vec<[u8; DIGEST_SIZE]>,
    output_write_accesses: Vec<U16Access>,
}

/// Generates the Deferral OUTPUT trace from immutable preflight history.
///
/// Raw output advice is recovered from its proof-visible logged writes. Replay
/// never reads deferral streams or invokes a host callback.
pub fn generate_trace_from_postflight<F: VmField>(
    chip: &DeferralOutputChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(DeferralOpcode::OUTPUT.global_opcode());
    let width = DeferralOutputCols::<F>::width();
    let mut rows_used = 0usize;
    let mut replay_sections = Vec::with_capacity(steps.len());

    // Validate and collect every section before mutating lookup producers.
    for &step in steps {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
        {
            return Err(PostflightError::new(
                "Deferral OUTPUT has invalid address spaces",
            ));
        }
        let deferral_idx = instruction.c.as_canonical_u32();
        if deferral_idx as usize >= chip.inner.count_chip.count.len() {
            return Err(PostflightError::new(
                "Deferral OUTPUT index is out of bounds",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs_ptr = instruction.b.as_canonical_u32();
        let mut replay = postflight.replay(step);
        let rd = replay.read_u16(
            RV64_REGISTER_AS,
            checked_u16_pointer(rd_ptr, "Deferral OUTPUT destination register")?,
        )?;
        let rs = replay.read_u16(
            RV64_REGISTER_AS,
            checked_u16_pointer(rs_ptr, "Deferral OUTPUT source register")?,
        )?;
        let rd_val = logged_u32_pointer(rd.value, "Deferral OUTPUT output pointer")?;
        let rs_val = logged_u32_pointer(rs.value, "Deferral OUTPUT input pointer")?;
        require_block_alignment(rd_val, "Deferral OUTPUT output pointer")?;
        require_block_alignment(rs_val, "Deferral OUTPUT input pointer")?;

        let mut output_key_accesses = Vec::with_capacity(OUTPUT_TOTAL_MEMORY_OPS);
        let mut output_key_bytes = Vec::with_capacity(OUTPUT_TOTAL_BYTES);
        for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
            let byte_pointer = checked_pointer_offset(
                rs_val,
                chunk_idx * MEMORY_BLOCK_BYTES,
                "Deferral OUTPUT input key pointer overflow",
            )?;
            let access = replay.read_u16(
                RV64_MEMORY_AS,
                checked_u16_pointer(byte_pointer, "Deferral OUTPUT input key pointer")?,
            )?;
            output_key_bytes.extend(rv64_u16_block_to_bytes(access.value));
            output_key_accesses.push(access);
        }
        let output_key: [u8; OUTPUT_TOTAL_BYTES] = output_key_bytes
            .try_into()
            .expect("OUTPUT_TOTAL_MEMORY_OPS covers OUTPUT_TOTAL_BYTES");
        let (output_commit, output_len_bytes) = split_output(output_key);
        let output_len_u64 = u64::from_le_bytes(output_len_bytes);
        let output_len = u32::try_from(output_len_u64)
            .map_err(|_| PostflightError::new("Deferral OUTPUT length exceeds u32"))?;
        if !output_len.is_multiple_of(DIGEST_SIZE as u32) {
            return Err(PostflightError::new(
                "Deferral OUTPUT length is not a whole sponge row",
            ));
        }
        let output_rows = usize::try_from(output_len / DIGEST_SIZE as u32)
            .map_err(|_| PostflightError::new("Deferral OUTPUT row count exceeds usize"))?;
        let num_rows = output_rows
            .checked_add(1)
            .ok_or_else(|| PostflightError::new("Deferral OUTPUT row count overflow"))?;
        rows_used = rows_used
            .checked_add(num_rows)
            .ok_or_else(|| PostflightError::new("Deferral OUTPUT trace height overflow"))?;

        let mut output_chunks: Vec<[u8; DIGEST_SIZE]> = Vec::with_capacity(output_rows);
        let write_capacity = output_rows
            .checked_mul(DIGEST_BYTE_MEMORY_OPS)
            .ok_or_else(|| PostflightError::new("Deferral OUTPUT write count overflow"))?;
        let mut output_write_accesses = Vec::with_capacity(write_capacity);
        for row_idx in 0..output_rows {
            let row_pointer = checked_pointer_offset(
                rd_val,
                row_idx * DIGEST_SIZE,
                "Deferral OUTPUT row pointer overflow",
            )?;
            let mut row_bytes = Vec::with_capacity(DIGEST_SIZE);
            for chunk_idx in 0..DIGEST_BYTE_MEMORY_OPS {
                let byte_pointer = checked_pointer_offset(
                    row_pointer,
                    chunk_idx * MEMORY_BLOCK_BYTES,
                    "Deferral OUTPUT chunk pointer overflow",
                )?;
                let access = replay.write_observed_u16(
                    RV64_MEMORY_AS,
                    checked_u16_pointer(byte_pointer, "Deferral OUTPUT chunk pointer")?,
                )?;
                row_bytes.extend(rv64_u16_block_to_bytes(access.value));
                output_write_accesses.push(access);
            }
            output_chunks.push(
                row_bytes
                    .try_into()
                    .expect("DIGEST_BYTE_MEMORY_OPS covers DIGEST_SIZE"),
            );
        }

        let mut initial = [F::ZERO; DIGEST_SIZE];
        initial[0] = F::from_u32(deferral_idx);
        initial[1] = F::from_u32(output_len);
        let mut current =
            chip.inner
                .poseidon2_chip
                .perm(&initial, &[F::ZERO; DIGEST_SIZE], num_rows == 1);
        for (row_idx, output_chunk) in output_chunks.iter().enumerate() {
            current = chip.inner.poseidon2_chip.perm(
                &output_chunk.map(F::from_u8),
                &current,
                row_idx + 1 == output_rows,
            );
        }
        if f_commit_to_bytes(&current) != output_commit {
            return Err(PostflightError::new(
                "Deferral OUTPUT logged bytes do not match its output commitment",
            ));
        }
        let next_pc = from_pc
            .checked_add(DEFAULT_PC_STEP)
            .ok_or_else(|| PostflightError::new("Deferral OUTPUT next PC overflow"))?;
        replay.finish(next_pc)?;
        replay_sections.push(DeferralOutputReplay {
            from_pc,
            from_timestamp,
            rd_ptr,
            rs_ptr,
            deferral_idx,
            rd_val,
            rs_val,
            rd,
            rs,
            output_commit,
            output_len,
            output_key_accesses,
            output_chunks,
            output_write_accesses,
        });
    }

    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();
    let mut trace_row = 0usize;
    for section in replay_sections {
        fill_output_section(
            &chip.inner,
            &mem_helper,
            &mut trace,
            &mut trace_row,
            section,
        );
    }
    debug_assert_eq!(trace_row, rows_used);
    Ok(trace)
}

fn fill_output_section<F: VmField>(
    filler: &DeferralOutputFiller<F>,
    mem_helper: &MemoryAuxColsFactory<F>,
    trace: &mut RowMajorMatrix<F>,
    trace_row: &mut usize,
    section: DeferralOutputReplay,
) {
    let width = trace.width;
    let output_rows = section.output_chunks.len();
    let num_rows = output_rows + 1;
    let mut initial_sponge_input = [F::ZERO; DIGEST_SIZE];
    initial_sponge_input[0] = F::from_u32(section.deferral_idx);
    initial_sponge_input[1] = F::from_u32(section.output_len);
    let output_len_bytes = section.output_len.to_le_bytes();
    let output_len_f = output_len_bytes.map(F::from_u8);
    let output_commit_f = section.output_commit.map(F::from_u8);
    let mut current_poseidon2_res = [F::ZERO; DIGEST_SIZE];
    filler.count_chip.add_count(section.deferral_idx);

    for row_idx in 0..num_rows {
        let row_start = *trace_row * width;
        *trace_row += 1;
        let cols: &mut DeferralOutputCols<F> =
            trace.values[row_start..row_start + width].borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(row_idx == 0);
        cols.is_last = F::from_bool(row_idx + 1 == num_rows);
        cols.section_idx = F::from_usize(row_idx);
        cols.from_state.pc = F::from_u32(section.from_pc);
        cols.from_state.timestamp = F::from_u32(section.from_timestamp);
        cols.rd_ptr = F::from_u32(section.rd_ptr);
        cols.rs_ptr = F::from_u32(section.rs_ptr);
        cols.deferral_idx = F::from_u32(section.deferral_idx);
        cols.rd_val = section.rd_val.to_le_bytes().map(F::from_u8);
        cols.rs_val = section.rs_val.to_le_bytes().map(F::from_u8);
        cols.output_len = output_len_f;
        cols.output_commit = output_commit_f;

        if row_idx == 0 {
            debug_assert!(RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS >= filler.address_bits);
            let limb_shift_bits = RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS - filler.address_bits;
            filler.bitwise_lookup_chip.request_range(
                (section.rd_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
                (section.rs_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
            );
            for pointer in [section.rd_val, section.rs_val] {
                for bytes in pointer.to_le_bytes().chunks_exact(2) {
                    filler
                        .bitwise_lookup_chip
                        .request_range(bytes[0] as u32, bytes[1] as u32);
                }
            }
            for bytes in output_len_bytes.chunks_exact(2) {
                filler
                    .bitwise_lookup_chip
                    .request_range(bytes[0] as u32, bytes[1] as u32);
            }
            filler.bitwise_lookup_chip.request_range(
                (output_len_bytes[F_NUM_BYTES - 1] as u32) << limb_shift_bits,
                0,
            );
            mem_helper.fill(
                section.rd.previous_timestamp,
                section.rd.timestamp,
                cols.rd_aux.as_mut(),
            );
            mem_helper.fill(
                section.rs.previous_timestamp,
                section.rs.timestamp,
                cols.rs_aux.as_mut(),
            );
            for (aux, access) in cols
                .output_commit_and_len_aux
                .iter_mut()
                .zip(&section.output_key_accesses)
            {
                mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
            }
            cols.sponge_inputs = initial_sponge_input;
            current_poseidon2_res = filler.poseidon2_chip.perm_and_record(
                &cols.sponge_inputs,
                &[F::ZERO; DIGEST_SIZE],
                num_rows == 1,
            );
        } else {
            let output_chunk = section.output_chunks[row_idx - 1];
            for bytes in output_chunk.chunks_exact(2) {
                filler
                    .bitwise_lookup_chip
                    .request_range(bytes[0] as u32, bytes[1] as u32);
            }
            cols.sponge_inputs = output_chunk.map(F::from_u8);
            current_poseidon2_res = filler.poseidon2_chip.perm_and_record(
                &cols.sponge_inputs,
                &current_poseidon2_res,
                row_idx + 1 == num_rows,
            );
            let write_start = (row_idx - 1) * DIGEST_BYTE_MEMORY_OPS;
            for (aux, access) in cols.write_bytes_aux.iter_mut().zip(
                &section.output_write_accesses[write_start..write_start + DIGEST_BYTE_MEMORY_OPS],
            ) {
                aux.set_prev_data(access.previous_value.map(F::from_u16));
                mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
            }
        }
        cols.poseidon2_res = current_poseidon2_res;
    }
    debug_assert_eq!(
        f_commit_to_bytes(&current_poseidon2_res),
        section.output_commit
    );
    for bytes in output_commit_f.chunks_exact(2) {
        filler
            .bitwise_lookup_chip
            .request_range(bytes[0].as_canonical_u32(), bytes[1].as_canonical_u32());
    }
    let first_row = (*trace_row - num_rows) * width;
    let cols: &mut DeferralOutputCols<F> = trace.values[first_row..first_row + width].borrow_mut();
    let output_commit_rcs = output_commit_f
        .chunks_exact(F_NUM_BYTES)
        .zip(cols.output_commit_lt_aux.iter_mut())
        .map(|(bytes, aux)| {
            let x_le = from_fn(|i| bytes[i]);
            CanonicityTraceGen::generate_subrow(&x_le, aux)
        })
        .collect_vec();
    for pair in output_commit_rcs.chunks_exact(2) {
        filler.bitwise_lookup_chip.request_range(pair[0], pair[1]);
    }
}

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
    pub write_aux: &'a mut [MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>],
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
            unsafe { rest.align_to_mut::<MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>>() };

        DeferralOutputRecordMut {
            header: header_buf.borrow_mut(),
            write_bytes,
            write_aux: &mut write_aux_buf[..num_write_rows * DIGEST_BYTE_MEMORY_OPS],
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
            total_len.next_multiple_of(align_of::<MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>>());
        total_len += num_write_rows
            * DIGEST_BYTE_MEMORY_OPS
            * size_of::<MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>>();
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
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    address_bits: usize,
}

impl<F, RA> PreflightExecutor<F, RA> for DeferralOutputExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, DeferralOutputLayout, DeferralOutputRecordMut<'buf>>,
{
    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { a, b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        let rd_ptr = a.as_canonical_u32();
        let rs_ptr = b.as_canonical_u32();
        let deferral_idx = c.as_canonical_u32();
        let deferral_state_idx =
            checked_deferral_index(*state.pc, state.streams.deferrals.len(), deferral_idx)?;

        // Do a non-tracing read to get the output_len and compute num_rows
        let read_ptr = read_rv64_register_as_u32(state.memory.data(), rs_ptr);
        let output_key_chunks: [[u8; MEMORY_BLOCK_BYTES]; OUTPUT_TOTAL_MEMORY_OPS] = from_fn(|i| {
            memory_read(
                state.memory.data(),
                RV64_MEMORY_AS,
                read_ptr + (i * MEMORY_BLOCK_BYTES) as u32,
            )
        });
        let output_key: [u8; OUTPUT_TOTAL_BYTES] = join_byte_memory_ops(output_key_chunks);
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
            tracing_read::<MEMORY_BLOCK_BYTES>(
                state.memory,
                RV64_MEMORY_AS,
                input_ptr + (chunk_idx * MEMORY_BLOCK_BYTES) as u32,
                &mut record.header.output_commit_and_len_aux[chunk_idx].prev_timestamp,
            );
        }

        let output_raw =
            state.streams.deferrals[deferral_state_idx].get_output(&output_commit.to_vec());
        debug_assert_eq!(output_raw.len(), output_len_val);

        for (row_idx, output_chunk) in output_raw.chunks_exact(DIGEST_SIZE).enumerate() {
            let row_output_ptr = output_ptr + (row_idx * DIGEST_SIZE) as u32;
            for chunk_idx in 0..DIGEST_BYTE_MEMORY_OPS {
                let aux_idx = row_idx * DIGEST_BYTE_MEMORY_OPS + chunk_idx;
                tracing_write(
                    state.memory,
                    RV64_MEMORY_AS,
                    row_output_ptr + (chunk_idx * MEMORY_BLOCK_BYTES) as u32,
                    byte_memory_op_chunk(output_chunk, chunk_idx),
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
                    debug_assert!(RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS >= self.address_bits);
                    let limb_shift_bits = RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS - self.address_bits;

                    self.bitwise_lookup_chip.request_range(
                        (header.rd_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32)
                            << limb_shift_bits,
                        (header.rs_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32)
                            << limb_shift_bits,
                    );
                    for ptr in [header.rd_val, header.rs_val] {
                        for bytes in ptr.to_le_bytes().chunks_exact(2) {
                            self.bitwise_lookup_chip
                                .request_range(bytes[0] as u32, bytes[1] as u32);
                        }
                    }
                    for bytes in output_len_bytes.chunks_exact(2) {
                        self.bitwise_lookup_chip
                            .request_range(bytes[0] as u32, bytes[1] as u32);
                    }
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
                    // The canonicity aux columns are only populated on the first row (the
                    // canonicity range check is gated by `is_first`). On non-first rows the
                    // preflight record may have left non-zero bytes in these columns, so clear
                    // them to satisfy the unconditional `assert_bool` constraints in the
                    // canonicity sub-AIR.
                    for aux in &mut cols.output_commit_lt_aux {
                        CanonicityTraceGen::clear_aux(aux);
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
                    for chunk_idx in 0..DIGEST_BYTE_MEMORY_OPS {
                        let aux_idx = (row_idx - 1) * DIGEST_BYTE_MEMORY_OPS + chunk_idx;
                        cols.write_bytes_aux[chunk_idx]
                            .set_prev_data(pack_u8_block_bytes(&write_aux[aux_idx].prev_data));
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
            for bytes in output_commit.chunks_exact(2) {
                self.bitwise_lookup_chip
                    .request_range(bytes[0].as_canonical_u32(), bytes[1].as_canonical_u32());
            }
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
