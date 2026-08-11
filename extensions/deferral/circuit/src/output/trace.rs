use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use itertools::Itertools;
use openvm_circuit::{
    arch::{Postflight, PostflightError, U16Access, VmField, MEMORY_BLOCK_BYTES},
    system::memory::MemoryAuxColsFactory,
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{BYTE_BITS, MEMORY_AS, REGISTER_AS, WORD_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::u16_block_to_bytes;
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    canonicity::CanonicityTraceGen,
    count::DeferralCircuitCountChip,
    output::{DeferralOutputChip, DeferralOutputCols},
    poseidon2::DeferralPoseidon2Chip,
    utils::{
        checked_pointer_offset, checked_u16_pointer, f_commit_to_bytes, logged_u32_pointer,
        require_block_alignment, split_output, DIGEST_BYTE_MEMORY_OPS, F_NUM_BYTES,
        OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
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
        if instruction.d.as_u32() != REGISTER_AS || instruction.e.as_u32() != MEMORY_AS {
            return Err(PostflightError::new(
                "Deferral OUTPUT has invalid address spaces",
            ));
        }
        let deferral_idx = instruction.c.as_u32();
        if deferral_idx as usize >= chip.inner.count_chip.count.len() {
            return Err(PostflightError::new(
                "Deferral OUTPUT index is out of bounds",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rd_ptr = instruction.a.as_u32();
        let rs_ptr = instruction.b.as_u32();
        let mut replay = postflight.replay(step);
        let rd = replay.read_u16(
            REGISTER_AS,
            checked_u16_pointer(rd_ptr, "Deferral OUTPUT destination register")?,
        )?;
        let rs = replay.read_u16(
            REGISTER_AS,
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
                MEMORY_AS,
                checked_u16_pointer(byte_pointer, "Deferral OUTPUT input key pointer")?,
            )?;
            output_key_bytes.extend(u16_block_to_bytes(access.value));
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
                    MEMORY_AS,
                    checked_u16_pointer(byte_pointer, "Deferral OUTPUT chunk pointer")?,
                )?;
                row_bytes.extend(u16_block_to_bytes(access.value));
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
            debug_assert!(BYTE_BITS * WORD_NUM_LIMBS >= filler.address_bits);
            let limb_shift_bits = BYTE_BITS * WORD_NUM_LIMBS - filler.address_bits;
            filler.bitwise_lookup_chip.request_range(
                (section.rd_val.to_le_bytes()[WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
                (section.rs_val.to_le_bytes()[WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
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

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputExecutor;

#[derive(Clone, derive_new::new)]
pub struct DeferralOutputFiller<F: VmField> {
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    address_bits: usize,
}
