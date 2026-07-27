use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{
        Postflight, PostflightError, PostflightReplay, PostflightStep, U16Access, BLOCK_FE_WIDTH,
    },
    system::memory::MemoryAuxColsFactory,
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64HintStoreOpcode::{HINT_BUFFER, HINT_STORED};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    validate_hint_buffer_num_words, Rv64HintStoreChip, Rv64HintStoreCols, REM_WORDS_SHIFT,
};
use crate::adapters::{
    byte_ptr_to_u16_ptr_value, ptr_bound_from_ptr, ptr_to_field_u16_limbs, RV64_PTR_BITS, U16_BITS,
};

struct HintStoreReplayInput {
    from_pc: u32,
    from_timestamp: u32,
    num_words: u32,
    mem_ptr_ptr: u32,
    num_words_ptr: Option<u32>,
    mem_ptr: u32,
    mem_ptr_read: U16Access,
    num_words_read: Option<U16Access>,
}

/// Generates the RV64 hint-store trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64HintStoreChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [HINT_STORED, HINT_BUFFER];
    let mut rows_used = 0usize;
    for opcode in opcodes {
        for &step in postflight.steps(opcode.global_opcode()) {
            let (_, input) = replay_header(postflight, step, chip.inner.pointer_max_bits)?;
            rows_used = rows_used
                .checked_add(input.num_words as usize)
                .ok_or_else(|| PostflightError::new("hint-store trace height overflow"))?;
        }
    }

    let width = Rv64HintStoreCols::<F>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();
    let mut row_index = 0usize;

    for opcode in opcodes {
        for &step in postflight.steps(opcode.global_opcode()) {
            let (mut replay, input) = replay_header(postflight, step, chip.inner.pointer_max_bits)?;
            chip.inner.range_checker_chip.add_count(
                ptr_bound_from_ptr(input.mem_ptr, chip.inner.pointer_max_bits),
                U16_BITS,
            );
            chip.inner
                .range_checker_chip
                .add_count(input.num_words << REM_WORDS_SHIFT, U16_BITS);

            for local_index in 0..input.num_words {
                if local_index != 0 {
                    replay.advance_timestamp(2)?;
                }
                let byte_ptr = input
                    .mem_ptr
                    .checked_add(local_index * RV64_REGISTER_NUM_LIMBS as u32)
                    .ok_or_else(|| PostflightError::new("hint-store memory pointer overflow"))?;
                let write = replay
                    .write_observed_u16(RV64_MEMORY_AS, byte_ptr_to_u16_ptr_value(byte_ptr))?;

                let row = &mut trace.values[row_index * width..(row_index + 1) * width];
                let cols: &mut Rv64HintStoreCols<F> = row.borrow_mut();
                fill_row(&mem_helper, cols, &input, local_index, byte_ptr, write);
                row_index += 1;
            }
            replay.finish(input.from_pc.wrapping_add(DEFAULT_PC_STEP))?;
        }
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
        || instruction.d.as_canonical_u32() != RV64_REGISTER_AS
        || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
        || instruction.f.as_canonical_u32() != 0
        || instruction.g.as_canonical_u32() != 0
    {
        return Err(PostflightError::new(
            "hint-store instruction has invalid fixed operands",
        ));
    }

    let mem_ptr_ptr = checked_register_pointer(instruction.b.as_canonical_u32(), "memory pointer")?;
    let num_words_ptr = if is_single {
        if instruction.a.as_canonical_u32() != 0 {
            return Err(PostflightError::new(
                "HINT_STORED must not specify a word-count register",
            ));
        }
        None
    } else {
        Some(checked_register_pointer(
            instruction.a.as_canonical_u32(),
            "word count",
        )?)
    };

    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let mut replay = postflight.replay(step);
    let mem_ptr_read = replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(mem_ptr_ptr))?;
    let mem_ptr_u64 = u16_block_to_u64(mem_ptr_read.value);
    let mem_ptr = u32::try_from(mem_ptr_u64)
        .ok()
        .filter(|pointer| pointer.is_multiple_of(RV64_REGISTER_NUM_LIMBS as u32))
        .ok_or_else(|| {
            PostflightError::new(
                "hint-store destination is not an aligned low-32-bit memory pointer",
            )
        })?;

    let (num_words, num_words_read) = if let Some(num_words_ptr) = num_words_ptr {
        let access = replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(num_words_ptr))?;
        let count = u16_block_to_u64(access.value);
        let count = validate_hint_buffer_num_words(from_pc, count)
            .map(u32::from)
            .map_err(|error| PostflightError::new(error.to_string()))?;
        (count, Some(access))
    } else {
        replay.advance_timestamp(1)?;
        (1, None)
    };

    let pointer_limit = if pointer_max_bits < RV64_PTR_BITS {
        1u64 << pointer_max_bits
    } else {
        1u64 << RV64_PTR_BITS
    };
    let access_end = u64::from(mem_ptr) + u64::from(num_words) * RV64_REGISTER_NUM_LIMBS as u64;
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
            mem_ptr_read,
            num_words_read,
        },
    ))
}

fn fill_row<F: PrimeField32>(
    mem_helper: &MemoryAuxColsFactory<F>,
    cols: &mut Rv64HintStoreCols<F>,
    input: &HintStoreReplayInput,
    local_index: u32,
    byte_ptr: u32,
    write: U16Access,
) {
    let is_single = input.num_words_ptr.is_none();
    let timestamp = input.from_timestamp + 3 * local_index;
    if local_index == 0 {
        mem_helper.fill(
            input.mem_ptr_read.previous_timestamp,
            input.mem_ptr_read.timestamp,
            cols.mem_ptr_aux_cols.as_mut(),
        );
    } else {
        mem_helper.fill_zero(cols.mem_ptr_aux_cols.as_mut());
    }
    if local_index == 0 {
        if let Some(num_words_read) = &input.num_words_read {
            mem_helper.fill(
                num_words_read.previous_timestamp,
                num_words_read.timestamp,
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
    cols.mem_ptr_ptr = F::from_u32(input.mem_ptr_ptr);
    cols.from_state.timestamp = F::from_u32(timestamp);
    cols.from_state.pc = F::from_u32(input.from_pc);
    cols.rem_words = F::from_u32(input.num_words - local_index);
    cols.is_buffer = F::from_bool(!is_single);
    cols.is_single = F::from_bool(is_single);
}

fn checked_register_pointer(pointer: u32, operand: &str) -> Result<u32, PostflightError> {
    if pointer >= 32 * RV64_REGISTER_NUM_LIMBS as u32
        || !pointer.is_multiple_of(RV64_REGISTER_NUM_LIMBS as u32)
    {
        return Err(PostflightError::new(format!(
            "hint-store {operand} is not an aligned register pointer"
        )));
    }
    Ok(pointer)
}

fn u16_block_to_u64(value: [u16; BLOCK_FE_WIDTH]) -> u64 {
    value
        .into_iter()
        .enumerate()
        .fold(0u64, |result, (index, limb)| {
            result | (u64::from(limb) << (index * U16_BITS))
        })
}
