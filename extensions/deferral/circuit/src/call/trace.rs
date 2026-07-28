use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use itertools::Itertools;
use openvm_circuit::{
    arch::{Postflight, PostflightError, U16Access, VmField, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::MemoryAuxColsFactory,
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_riscv_circuit::adapters::rv64_u16_block_to_bytes;
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::{accumulator_ptrs, DeferralCallChip};
use crate::{
    call::{DeferralCallAdapterCols, DeferralCallCoreCols},
    canonicity::CanonicityTraceGen,
    count::DeferralCircuitCountChip,
    poseidon2::DeferralPoseidon2Chip,
    utils::{
        byte_commit_to_f, checked_pointer_offset, checked_u16_pointer, f_memory_op_chunk,
        logged_u32_pointer, require_block_alignment, COMMIT_MEMORY_OPS, COMMIT_NUM_BYTES,
        DIGEST_F_MEMORY_OPS, F_NUM_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
    DeferralFn,
};

struct DeferralCallReplay<F> {
    from_pc: u32,
    from_timestamp: u32,
    rd_ptr: u32,
    rs_ptr: u32,
    rd_val: u32,
    rs_val: u32,
    deferral_idx: u32,
    rd: U16Access,
    rs: U16Access,
    input_commit: [u8; COMMIT_NUM_BYTES],
    input_commit_accesses: Vec<U16Access>,
    old_input_acc: [F; DIGEST_SIZE],
    old_input_acc_accesses: Vec<FieldAccessReplay<F>>,
    old_output_acc: [F; DIGEST_SIZE],
    old_output_acc_accesses: Vec<FieldAccessReplay<F>>,
    output_commit: [u8; COMMIT_NUM_BYTES],
    output_len: [u8; F_NUM_BYTES],
    output_accesses: Vec<U16Access>,
    new_input_acc: [F; DIGEST_SIZE],
    new_input_acc_accesses: Vec<FieldAccessReplay<F>>,
    new_output_acc: [F; DIGEST_SIZE],
    new_output_acc_accesses: Vec<FieldAccessReplay<F>>,
}

struct FieldAccessReplay<F> {
    previous_value: [F; BLOCK_FE_WIDTH],
    previous_timestamp: u32,
    timestamp: u32,
}

/// Generates the Deferral CALL trace from immutable preflight history.
///
/// The host deferral function is deliberately not called here. Its output key
/// is recovered from the ordinary proof-visible heap writes made by serial
/// preflight, while the accumulator updates are recomputed deterministically.
pub fn generate_trace_from_postflight<F: VmField>(
    chip: &DeferralCallChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(DeferralOpcode::CALL.global_opcode());
    let adapter_width = DeferralCallAdapterCols::<F>::width();
    let width = adapter_width + DeferralCallCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mut replay_rows = Vec::with_capacity(steps.len());

    // Validate the complete history before mutating any lookup producer.
    for &step in steps {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
        {
            return Err(PostflightError::new(
                "Deferral CALL has invalid address spaces",
            ));
        }
        let deferral_idx = instruction.c.as_canonical_u32();
        if deferral_idx as usize >= chip.inner.count_chip.count.len() {
            return Err(PostflightError::new("Deferral CALL index is out of bounds"));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs_ptr = instruction.b.as_canonical_u32();
        let mut replay = postflight.replay(step);

        let rd = replay.read_u16(
            RV64_REGISTER_AS,
            checked_u16_pointer(rd_ptr, "Deferral CALL destination register")?,
        )?;
        let rs = replay.read_u16(
            RV64_REGISTER_AS,
            checked_u16_pointer(rs_ptr, "Deferral CALL source register")?,
        )?;
        let rd_val = logged_u32_pointer(rd.value, "Deferral CALL output pointer")?;
        let rs_val = logged_u32_pointer(rs.value, "Deferral CALL input pointer")?;
        require_block_alignment(rd_val, "Deferral CALL output pointer")?;
        require_block_alignment(rs_val, "Deferral CALL input pointer")?;

        let mut input_commit_accesses = Vec::with_capacity(COMMIT_MEMORY_OPS);
        let mut input_commit_bytes = Vec::with_capacity(COMMIT_NUM_BYTES);
        for chunk_idx in 0..COMMIT_MEMORY_OPS {
            let byte_pointer = checked_pointer_offset(
                rs_val,
                chunk_idx * MEMORY_BLOCK_BYTES,
                "Deferral CALL input commit pointer overflow",
            )?;
            let access = replay.read_u16(
                RV64_MEMORY_AS,
                checked_u16_pointer(byte_pointer, "Deferral CALL input commit pointer")?,
            )?;
            input_commit_bytes.extend(rv64_u16_block_to_bytes(access.value));
            input_commit_accesses.push(access);
        }
        let input_commit: [u8; COMMIT_NUM_BYTES] = input_commit_bytes
            .try_into()
            .expect("COMMIT_MEMORY_OPS covers COMMIT_NUM_BYTES");

        let (input_acc_ptr, output_acc_ptr) = accumulator_ptrs(deferral_idx);
        let mut old_input_acc_accesses = Vec::with_capacity(DIGEST_F_MEMORY_OPS);
        let mut old_input_acc_values = Vec::with_capacity(DIGEST_SIZE);
        for chunk_idx in 0..DIGEST_F_MEMORY_OPS {
            let pointer = checked_pointer_offset(
                input_acc_ptr,
                chunk_idx * BLOCK_FE_WIDTH,
                "Deferral CALL input accumulator pointer overflow",
            )?;
            let access = replay.read_field32(DEFERRAL_AS, pointer)?;
            old_input_acc_values.extend(access.value);
            old_input_acc_accesses.push(FieldAccessReplay {
                previous_value: access.previous_value,
                previous_timestamp: access.previous_timestamp,
                timestamp: access.timestamp,
            });
        }
        let old_input_acc = old_input_acc_values
            .try_into()
            .expect("DIGEST_F_MEMORY_OPS covers DIGEST_SIZE");

        let mut old_output_acc_accesses = Vec::with_capacity(DIGEST_F_MEMORY_OPS);
        let mut old_output_acc_values = Vec::with_capacity(DIGEST_SIZE);
        for chunk_idx in 0..DIGEST_F_MEMORY_OPS {
            let pointer = checked_pointer_offset(
                output_acc_ptr,
                chunk_idx * BLOCK_FE_WIDTH,
                "Deferral CALL output accumulator pointer overflow",
            )?;
            let access = replay.read_field32(DEFERRAL_AS, pointer)?;
            old_output_acc_values.extend(access.value);
            old_output_acc_accesses.push(FieldAccessReplay {
                previous_value: access.previous_value,
                previous_timestamp: access.previous_timestamp,
                timestamp: access.timestamp,
            });
        }
        let old_output_acc = old_output_acc_values
            .try_into()
            .expect("DIGEST_F_MEMORY_OPS covers DIGEST_SIZE");

        let mut output_accesses = Vec::with_capacity(OUTPUT_TOTAL_MEMORY_OPS);
        let mut output_bytes = Vec::with_capacity(crate::utils::OUTPUT_TOTAL_BYTES);
        for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
            let byte_pointer = checked_pointer_offset(
                rd_val,
                chunk_idx * MEMORY_BLOCK_BYTES,
                "Deferral CALL output key pointer overflow",
            )?;
            let access = replay.write_observed_u16(
                RV64_MEMORY_AS,
                checked_u16_pointer(byte_pointer, "Deferral CALL output key pointer")?,
            )?;
            output_bytes.extend(rv64_u16_block_to_bytes(access.value));
            output_accesses.push(access);
        }
        let output_bytes: [u8; crate::utils::OUTPUT_TOTAL_BYTES] = output_bytes
            .try_into()
            .expect("OUTPUT_TOTAL_MEMORY_OPS covers OUTPUT_TOTAL_BYTES");
        let (output_commit, output_len_full) = crate::utils::split_output(output_bytes);
        if output_len_full[F_NUM_BYTES..] != [0; crate::utils::OUTPUT_LEN_NUM_BYTES - F_NUM_BYTES] {
            return Err(PostflightError::new(
                "Deferral CALL logged a nonzero high output-length word",
            ));
        }
        let output_len = output_len_full[..F_NUM_BYTES]
            .try_into()
            .expect("slice has F_NUM_BYTES elements");

        let input_f_commit = byte_commit_to_f(&input_commit.map(F::from_u8));
        let output_f_commit = byte_commit_to_f(&output_commit.map(F::from_u8));
        let new_input_acc = chip
            .inner
            .poseidon2_chip
            .perm(&old_input_acc, &input_f_commit, true);
        let new_output_acc =
            chip.inner
                .poseidon2_chip
                .perm(&old_output_acc, &output_f_commit, true);

        let mut new_input_acc_accesses = Vec::with_capacity(DIGEST_F_MEMORY_OPS);
        for chunk_idx in 0..DIGEST_F_MEMORY_OPS {
            let pointer = checked_pointer_offset(
                input_acc_ptr,
                chunk_idx * BLOCK_FE_WIDTH,
                "Deferral CALL input accumulator pointer overflow",
            )?;
            let access = replay.write_field32(
                DEFERRAL_AS,
                pointer,
                f_memory_op_chunk(&new_input_acc, chunk_idx),
            )?;
            new_input_acc_accesses.push(FieldAccessReplay {
                previous_value: access.previous_value,
                previous_timestamp: access.previous_timestamp,
                timestamp: access.timestamp,
            });
        }
        let mut new_output_acc_accesses = Vec::with_capacity(DIGEST_F_MEMORY_OPS);
        for chunk_idx in 0..DIGEST_F_MEMORY_OPS {
            let pointer = checked_pointer_offset(
                output_acc_ptr,
                chunk_idx * BLOCK_FE_WIDTH,
                "Deferral CALL output accumulator pointer overflow",
            )?;
            let access = replay.write_field32(
                DEFERRAL_AS,
                pointer,
                f_memory_op_chunk(&new_output_acc, chunk_idx),
            )?;
            new_output_acc_accesses.push(FieldAccessReplay {
                previous_value: access.previous_value,
                previous_timestamp: access.previous_timestamp,
                timestamp: access.timestamp,
            });
        }
        let next_pc = from_pc
            .checked_add(DEFAULT_PC_STEP)
            .ok_or_else(|| PostflightError::new("Deferral CALL next PC overflow"))?;
        replay.finish(next_pc)?;

        replay_rows.push(DeferralCallReplay {
            from_pc,
            from_timestamp,
            rd_ptr,
            rs_ptr,
            rd_val,
            rs_val,
            deferral_idx,
            rd,
            rs,
            input_commit,
            input_commit_accesses,
            old_input_acc,
            old_input_acc_accesses,
            old_output_acc,
            old_output_acc_accesses,
            output_commit,
            output_len,
            output_accesses,
            new_input_acc,
            new_input_acc_accesses,
            new_output_acc,
            new_output_acc_accesses,
        });
    }

    let mem_helper = chip.mem_helper.as_borrowed();
    for (row_idx, replay_row) in replay_rows.into_iter().enumerate() {
        let row = &mut trace.values[row_idx * width..(row_idx + 1) * width];
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let adapter_row: &mut DeferralCallAdapterCols<F> = adapter_row.borrow_mut();
        fill_call_adapter(&chip.inner.adapter, &mem_helper, adapter_row, &replay_row);
        fill_call_core(&chip.inner, core_row.borrow_mut(), &replay_row);
    }
    Ok(trace)
}

fn fill_call_adapter<F: VmField>(
    filler: &DeferralCallAdapterFiller,
    mem_helper: &MemoryAuxColsFactory<F>,
    cols: &mut DeferralCallAdapterCols<F>,
    replay: &DeferralCallReplay<F>,
) {
    debug_assert!(RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS >= filler.address_bits);
    let limb_shift_bits = RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS - filler.address_bits;
    filler.bitwise_lookup_chip.request_range(
        (replay.rd_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
        (replay.rs_val.to_le_bytes()[RV64_WORD_NUM_LIMBS - 1] as u32) << limb_shift_bits,
    );
    for pointer in [replay.rd_val, replay.rs_val] {
        for bytes in pointer.to_le_bytes().chunks_exact(2) {
            filler
                .bitwise_lookup_chip
                .request_range(bytes[0] as u32, bytes[1] as u32);
        }
    }

    for (aux, access) in cols
        .new_output_acc_aux
        .iter_mut()
        .zip(&replay.new_output_acc_accesses)
    {
        aux.set_prev_data(access.previous_value);
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    for (aux, access) in cols
        .new_input_acc_aux
        .iter_mut()
        .zip(&replay.new_input_acc_accesses)
    {
        aux.set_prev_data(access.previous_value);
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    for (aux, access) in cols
        .output_commit_and_len_aux
        .iter_mut()
        .zip(&replay.output_accesses)
    {
        aux.set_prev_data(access.previous_value.map(F::from_u16));
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    for (aux, access) in cols
        .old_output_acc_aux
        .iter_mut()
        .zip(&replay.old_output_acc_accesses)
    {
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    for (aux, access) in cols
        .old_input_acc_aux
        .iter_mut()
        .zip(&replay.old_input_acc_accesses)
    {
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    for (aux, access) in cols
        .input_commit_aux
        .iter_mut()
        .zip(&replay.input_commit_accesses)
    {
        mem_helper.fill(access.previous_timestamp, access.timestamp, aux.as_mut());
    }
    mem_helper.fill(
        replay.rs.previous_timestamp,
        replay.rs.timestamp,
        cols.rs_aux.as_mut(),
    );
    mem_helper.fill(
        replay.rd.previous_timestamp,
        replay.rd.timestamp,
        cols.rd_aux.as_mut(),
    );
    cols.rs_val = replay.rs_val.to_le_bytes().map(F::from_u8);
    cols.rd_val = replay.rd_val.to_le_bytes().map(F::from_u8);
    cols.rs_ptr = F::from_u32(replay.rs_ptr);
    cols.rd_ptr = F::from_u32(replay.rd_ptr);
    cols.from_state.timestamp = F::from_u32(replay.from_timestamp);
    cols.from_state.pc = F::from_u32(replay.from_pc);
}

fn fill_call_core<F: VmField>(
    filler: &DeferralCallCoreFiller<DeferralCallAdapterFiller, F>,
    cols: &mut DeferralCallCoreCols<F>,
    replay: &DeferralCallReplay<F>,
) {
    filler.count_chip.add_count(replay.deferral_idx);
    let recorded_input_acc = filler.poseidon2_chip.perm_and_record(
        &replay.old_input_acc,
        &byte_commit_to_f(&replay.input_commit.map(F::from_u8)),
        true,
    );
    debug_assert_eq!(recorded_input_acc, replay.new_input_acc);
    let recorded_output_acc = filler.poseidon2_chip.perm_and_record(
        &replay.old_output_acc,
        &byte_commit_to_f(&replay.output_commit.map(F::from_u8)),
        true,
    );
    debug_assert_eq!(recorded_output_acc, replay.new_output_acc);
    for bytes in replay.output_commit.chunks_exact(2) {
        filler
            .bitwise_lookup_chip
            .request_range(bytes[0] as u32, bytes[1] as u32);
    }
    for bytes in replay.input_commit.chunks_exact(2) {
        filler
            .bitwise_lookup_chip
            .request_range(bytes[0] as u32, bytes[1] as u32);
    }
    for bytes in replay.output_len.chunks_exact(2) {
        filler
            .bitwise_lookup_chip
            .request_range(bytes[0] as u32, bytes[1] as u32);
    }
    debug_assert!(RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS >= filler.address_bits);
    let limb_shift_bits = RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS - filler.address_bits;
    filler.bitwise_lookup_chip.request_range(
        (replay.output_len[F_NUM_BYTES - 1] as u32) << limb_shift_bits,
        0,
    );

    let input_commit_f = replay.input_commit.map(F::from_u8);
    let output_commit_f = replay.output_commit.map(F::from_u8);
    let input_commit_rcs = input_commit_f
        .chunks_exact(F_NUM_BYTES)
        .zip(cols.input_commit_lt_aux.iter_mut())
        .map(|(bytes, aux)| {
            let x_le = from_fn(|i| bytes[i]);
            CanonicityTraceGen::generate_subrow(&x_le, aux)
        })
        .collect_vec();
    for pair in input_commit_rcs.chunks_exact(2) {
        filler.bitwise_lookup_chip.request_range(pair[0], pair[1]);
    }
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
    cols.is_valid = F::ONE;
    cols.deferral_idx = F::from_u32(replay.deferral_idx);
    cols.reads.input_commit = input_commit_f;
    cols.reads.old_input_acc = replay.old_input_acc;
    cols.reads.old_output_acc = replay.old_output_acc;
    cols.writes.output_commit = output_commit_f;
    cols.writes.output_len = replay.output_len.map(F::from_u8);
    cols.writes.new_input_acc = replay.new_input_acc;
    cols.writes.new_output_acc = replay.new_output_acc;
}

// ========================= CORE ==============================

#[derive(Clone, derive_new::new)]
pub struct DeferralCallCoreExecutor {
    pub(in crate::call) deferral_fns: Vec<Arc<DeferralFn>>,
}

#[derive(Clone, derive_new::new)]
pub struct DeferralCallCoreFiller<A, F: VmField> {
    adapter: A,
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    address_bits: usize,
}

// ========================= ADAPTER ==============================

#[derive(Clone, derive_new::new)]
pub struct DeferralCallAdapterFiller {
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    address_bits: usize,
}
