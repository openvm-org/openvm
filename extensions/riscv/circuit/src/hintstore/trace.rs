use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{
        Postflight, PostflightError, PostflightReplay, PostflightStep, U16Access, BLOCK_FE_WIDTH,
    },
    system::memory::MemoryAuxColsFactory,
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::{pc_to_idx, DEFAULT_PC_STEP},
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::HintStoreOpcode::{HINT_BUFFER, HINT_STORED};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{validate_hint_buffer_num_words, HintStoreChip, HintStoreCols, REM_WORDS_SHIFT};
use crate::adapters::{
    byte_ptr_limbs_to_cell_ptr_limbs_value, byte_ptr_to_u16_ptr_value, cell_ptr_hi_bits,
    checked_register_pointer, ptr_to_field_u16_limbs, u32_to_ptr_limbs, PTR_BITS, U16_BITS,
};

struct HintStoreReplayInput {
    from_pc: u32,
    from_timestamp: u32,
    num_words: u32,
    mem_ptr_ptr: u32,
    num_words_ptr: Option<u32>,
    mem_ptr: u32,
    mem_ptr_prev_timestamp: u32,
    num_words_prev_timestamp: Option<u32>,
}

/// Generates the RV64 hint-store trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &HintStoreChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [HINT_STORED, HINT_BUFFER];
    let mut rows_used = 0usize;
    let mut replay_inputs = Vec::with_capacity(
        opcodes
            .iter()
            .map(|opcode| postflight.steps(opcode.global_opcode()).len())
            .sum(),
    );
    for opcode in opcodes {
        for &step in postflight.steps(opcode.global_opcode()) {
            let replay_input = replay_header(postflight, step, chip.inner.pointer_max_bits)?;
            rows_used = rows_used
                .checked_add(replay_input.1.num_words as usize)
                .ok_or_else(|| PostflightError::new("hint-store trace height overflow"))?;
            replay_inputs.push(replay_input);
        }
    }

    let width = HintStoreCols::<F>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();
    let mut row_index = 0usize;

    for (mut replay, input) in replay_inputs {
        // Range check num_words (using MAX_HINT_BUFFER_DWORDS_BITS). Per-row pointer-limb
        // range checks are added in `fill_row`.
        chip.inner
            .range_checker_chip
            .add_count(input.num_words << REM_WORDS_SHIFT, U16_BITS);

        for local_index in 0..input.num_words {
            if local_index != 0 {
                replay.advance_timestamp(2)?;
            }
            let byte_ptr = input
                .mem_ptr
                .checked_add(local_index * REGISTER_NUM_LIMBS as u32)
                .ok_or_else(|| PostflightError::new("hint-store memory pointer overflow"))?;
            let write =
                replay.write_observed_u16(MEMORY_AS, byte_ptr_to_u16_ptr_value(byte_ptr))?;

            let row = &mut trace.values[row_index * width..(row_index + 1) * width];
            let cols: &mut HintStoreCols<F> = row.borrow_mut();
            fill_row(
                &chip.inner.range_checker_chip,
                chip.inner.pointer_max_bits,
                &mem_helper,
                cols,
                &input,
                local_index,
                byte_ptr,
                write,
            );
            row_index += 1;
        }
        replay.finish(input.from_pc.wrapping_add(DEFAULT_PC_STEP))?;
    }

    Ok(trace)
}

fn replay_header<'postflight, 'history, F: PrimeField32>(
    postflight: &'postflight Postflight<'history, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
) -> Result<
    (
        PostflightReplay<'postflight, 'history, F>,
        HintStoreReplayInput,
    ),
    PostflightError,
> {
    let instruction = postflight.instruction(step);
    let opcode = instruction.opcode;
    let is_single = opcode == HINT_STORED.global_opcode();
    if !is_single && opcode != HINT_BUFFER.global_opcode() {
        return Err(PostflightError::new(
            "hint-store replay received an unsupported opcode",
        ));
    }
    if instruction.c.as_canonical_u32() != 0
        || instruction.d.as_canonical_u32() != REGISTER_AS
        || instruction.e.as_canonical_u32() != MEMORY_AS
        || instruction.f.as_canonical_u32() != 0
        || instruction.g.as_canonical_u32() != 0
    {
        return Err(PostflightError::new(
            "hint-store instruction has invalid fixed operands",
        ));
    }

    let mem_ptr_ptr = u32::from(checked_register_pointer(instruction.b.as_canonical_u32())?);
    let num_words_ptr = if is_single {
        if instruction.a.as_canonical_u32() != 0 {
            return Err(PostflightError::new(
                "HINT_STORED must not specify a word-count register",
            ));
        }
        None
    } else {
        Some(u32::from(checked_register_pointer(
            instruction.a.as_canonical_u32(),
        )?))
    };

    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let mut replay = postflight.replay(step);
    let mem_ptr_read = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(mem_ptr_ptr))?;
    let mem_ptr_u64 = u16_block_to_u64(mem_ptr_read.value);
    let mem_ptr = u32::try_from(mem_ptr_u64)
        .ok()
        .filter(|pointer| pointer.is_multiple_of(REGISTER_NUM_LIMBS as u32))
        .ok_or_else(|| {
            PostflightError::new(
                "hint-store destination is not an aligned low-32-bit memory pointer",
            )
        })?;

    let (num_words, num_words_prev_timestamp) = if let Some(num_words_ptr) = num_words_ptr {
        let access = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(num_words_ptr))?;
        let count = u16_block_to_u64(access.value);
        let count = validate_hint_buffer_num_words(from_pc, count)
            .map(u32::from)
            .map_err(|error| PostflightError::new(error.to_string()))?;
        (count, Some(access.previous_timestamp))
    } else {
        replay.advance_timestamp(1)?;
        (1, None)
    };

    let pointer_limit = if pointer_max_bits < PTR_BITS {
        1u64 << pointer_max_bits
    } else {
        1u64 << PTR_BITS
    };
    let access_end = u64::from(mem_ptr) + u64::from(num_words) * REGISTER_NUM_LIMBS as u64;
    if access_end > pointer_limit {
        return Err(PostflightError::new(
            "hint-store output exceeds the implemented memory address space",
        ));
    }

    Ok((
        replay,
        HintStoreReplayInput {
            from_pc,
            from_timestamp,
            num_words,
            mem_ptr_ptr,
            num_words_ptr,
            mem_ptr,
            mem_ptr_prev_timestamp: mem_ptr_read.previous_timestamp,
            num_words_prev_timestamp,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn fill_row<F: PrimeField32>(
    range_checker: &openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
    mem_helper: &MemoryAuxColsFactory<F>,
    cols: &mut HintStoreCols<F>,
    input: &HintStoreReplayInput,
    local_index: u32,
    byte_ptr: u32,
    write: U16Access,
) {
    let is_single = input.num_words_ptr.is_none();
    let timestamp = input.from_timestamp + 3 * local_index;
    if local_index == 0 {
        mem_helper.fill(
            input.mem_ptr_prev_timestamp,
            input.from_timestamp,
            cols.mem_ptr_aux_cols.as_mut(),
        );
    } else {
        mem_helper.fill_zero(cols.mem_ptr_aux_cols.as_mut());
    }
    if local_index == 0 {
        if let Some(previous_timestamp) = input.num_words_prev_timestamp {
            mem_helper.fill(
                previous_timestamp,
                input.from_timestamp + 1,
                cols.num_words_aux_cols.as_mut(),
            );
            cols.num_words_ptr = F::from_u32(input.num_words_ptr.unwrap());
        } else {
            mem_helper.fill_zero(cols.num_words_aux_cols.as_mut());
        }
    } else {
        mem_helper.fill_zero(cols.num_words_aux_cols.as_mut());
    }

    cols.write_aux
        .set_prev_data(write.previous_value.map(F::from_u16));
    mem_helper.fill(
        write.previous_timestamp,
        write.timestamp,
        cols.write_aux.as_mut(),
    );
    cols.data = write.value.map(F::from_u16);
    cols.is_buffer_start = F::from_bool(local_index == 0 && !is_single);
    cols.mem_ptr_limbs = ptr_to_field_u16_limbs(byte_ptr);
    // Byte -> cell pointer conversion (heap write) and the per-row range checks: cell_hi
    // (hi_bits) and the low byte limb (16 bits, for the limb-wise `+8` increment).
    let byte_limbs = u32_to_ptr_limbs(byte_ptr);
    let (mem_carry, cell_limbs) = byte_ptr_limbs_to_cell_ptr_limbs_value(byte_limbs);
    cols.mem_ptr_carry = F::from_u32(mem_carry);
    // `+8` carry from this row's low byte limb into the high limb.
    cols.mem_ptr_inc_carry = F::from_u32((byte_limbs[0] + REGISTER_NUM_LIMBS as u32) >> U16_BITS);
    range_checker.add_count(cell_limbs[1], cell_ptr_hi_bits(pointer_max_bits));
    range_checker.add_count(byte_limbs[0], U16_BITS);
    cols.mem_ptr_ptr = F::from_u32(input.mem_ptr_ptr);
    cols.from_state.timestamp = F::from_u32(timestamp);
    cols.from_state.pc = F::from_u32(pc_to_idx(input.from_pc));
    cols.rem_words = F::from_u32(input.num_words - local_index);
    cols.is_buffer = F::from_bool(!is_single);
    cols.is_single = F::from_bool(is_single);
}

fn u16_block_to_u64(value: [u16; BLOCK_FE_WIDTH]) -> u64 {
    value
        .into_iter()
        .enumerate()
        .fold(0u64, |result, (index, limb)| {
            result | (u64::from(limb) << (index * U16_BITS))
        })
}
