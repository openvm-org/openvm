use core::convert::TryInto;
use std::{
    borrow::BorrowMut,
    sync::{atomic::Ordering, Arc, Mutex},
};

use openvm_circuit::{
    arch::{
        Postflight, PostflightError, PostflightStep, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
        U16_CELL_SIZE,
    },
    system::memory::{MemoryAuxColsFactory, SharedMemoryHelper},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerChip},
    U16_BITS,
};
use openvm_instructions::{
    program::{pc_to_limbs, DEFAULT_PC_STEP},
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::{
    add_const_u16_limbs_value, byte_ptr_limbs_to_cell_ptr_limbs_value, byte_ptr_to_u16_ptr_value,
    bytes_to_u16_block, cell_ptr_hi_bits, ptr_to_field_u16_limbs, try_bytes_to_u32,
    u16_block_to_bytes, u32_to_ptr_limbs,
};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use super::NUM_OP_ROWS_PER_INS;
use crate::{
    keccakf_op::{columns::KeccakfOpCols, keccakf_postimage_bytes},
    KECCAK_WIDTH_BYTES, KECCAK_WIDTH_MEM_OPS,
};

#[derive(derive_new::new)]
pub struct KeccakfOpChip<F> {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    pub(crate) shared_preimages: Arc<Mutex<Vec<KeccakfPreimage>>>,
}

struct KeccakfReplay {
    pc: u32,
    timestamp: u32,
    rd_ptr: u32,
    buffer_ptr: u32,
    rd_prev_timestamp: u32,
    buffer_prev_timestamps: [u32; KECCAK_WIDTH_MEM_OPS],
    preimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
    postimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
}

#[derive(Clone)]
pub(crate) struct KeccakfPreimage {
    pub timestamp: u32,
    pub bytes: [u8; KECCAK_WIDTH_BYTES],
}

impl<F: PrimeField32> KeccakfOpChip<F> {
    fn fill_trace_row(
        &self,
        range_checker: &VariableRangeCheckerChip,
        mem_helper: &MemoryAuxColsFactory<F>,
        row: &mut [F],
        replay: &KeccakfReplay,
    ) {
        row.fill(F::ZERO);
        let local: &mut KeccakfOpCols<F> = row.borrow_mut();

        local.pc = pc_to_limbs(replay.pc).map(F::from_u32);
        local.is_valid = F::ONE;
        local.timestamp = F::from_u32(replay.timestamp);
        local.rd_ptr = F::from_u32(replay.rd_ptr);
        local.buffer_ptr_limbs = ptr_to_field_u16_limbs(replay.buffer_ptr);

        // Pack consecutive pairs of state bytes into u16 cells.
        for (dst, bytes) in local
            .preimage
            .iter_mut()
            .zip(replay.preimage_buffer_bytes.chunks_exact(2))
        {
            *dst = F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        for (dst, bytes) in local
            .postimage
            .iter_mut()
            .zip(replay.postimage_buffer_bytes.chunks_exact(2))
        {
            *dst = F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]]));
        }

        let mut timestamp = replay.timestamp;
        mem_helper.fill(
            replay.rd_prev_timestamp,
            replay.timestamp,
            local.rd_aux.as_mut(),
        );
        timestamp += 1;
        for (aux, &previous_timestamp) in local
            .buffer_word_aux
            .iter_mut()
            .zip(&replay.buffer_prev_timestamps)
        {
            mem_helper.fill(previous_timestamp, timestamp, aux);
            timestamp += 1;
        }

        // Byte -> cell pointer conversion carry and per-block cell-offset carries, plus the
        // matching range-check counts (one `cell_hi` count for the conversion, one 16-bit
        // low-limb count per block), mirroring the AIR's per-valid-row multiplicities.
        let cell_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;
        let (conv_carry, base_cell) =
            byte_ptr_limbs_to_cell_ptr_limbs_value(u32_to_ptr_limbs(replay.buffer_ptr));
        range_checker.add_count(base_cell[1], cell_ptr_hi_bits(self.pointer_max_bits));
        local.buffer_cell_carry = F::from_u32(conv_carry);
        for (j, col) in local.buffer_word_add_carry.iter_mut().enumerate() {
            let (add_carry, block_cell_ptr) =
                add_const_u16_limbs_value(base_cell, j as u32 * cell_stride);
            range_checker.add_count(block_cell_ptr[0], U16_BITS);
            *col = F::from_u32(add_carry);
        }
    }
}

/// Generates the KeccakF operation trace directly from immutable preflight history.
pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &KeccakfOpChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(KeccakfOpcode::KECCAKF.global_opcode());
    let width = KeccakfOpCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len() * NUM_OP_ROWS_PER_INS);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let temporary_range_checker =
        Arc::new(VariableRangeCheckerChip::new(chip.range_checker_chip.bus()));
    let temporary_mem_helper = SharedMemoryHelper::new(
        temporary_range_checker.clone(),
        chip.mem_helper.timestamp_max_bits(),
    );
    let mem_helper = temporary_mem_helper.as_borrowed();
    let preimages = trace.values[..steps.len() * width * NUM_OP_ROWS_PER_INS]
        .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
        .zip(steps.par_iter().copied())
        .map(|(row, step)| {
            let replay = replay_step(postflight, step, chip.pointer_max_bits)?;
            chip.fill_trace_row(temporary_range_checker.as_ref(), &mem_helper, row, &replay);
            Ok(KeccakfPreimage {
                timestamp: replay.timestamp,
                bytes: replay.preimage_buffer_bytes,
            })
        })
        .collect::<Result<Vec<_>, PostflightError>>()?;
    if chip.range_checker_chip.count.len() != temporary_range_checker.count.len() {
        return Err(PostflightError::new("KECCAKF range-checker shape mismatch"));
    }
    for (destination, source) in chip
        .range_checker_chip
        .count
        .iter()
        .zip(&temporary_range_checker.count)
    {
        destination.fetch_add(source.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    *chip.shared_preimages.lock().unwrap() = preimages;
    Ok(trace)
}

fn replay_step<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
) -> Result<KeccakfReplay, PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.b != F::ZERO
        || instruction.c != F::ZERO
        || instruction.d.as_canonical_u32() != REGISTER_AS
        || instruction.e.as_canonical_u32() != MEMORY_AS
    {
        return Err(PostflightError::new(
            "KECCAKF instruction has invalid operands",
        ));
    }

    let pc = postflight.pc(step);
    let timestamp = postflight.timestamp(step);
    let rd_ptr = instruction.a.as_canonical_u32();
    if rd_ptr & 1 != 0 {
        return Err(PostflightError::new(
            "KECCAKF register pointer must be two-byte aligned",
        ));
    }
    let mut replay = postflight.replay(step);
    let rd = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(rd_ptr))?;
    let buffer_ptr = try_bytes_to_u32(u16_block_to_bytes(rd.value))
        .ok_or_else(|| PostflightError::new("KECCAKF buffer pointer exceeds 32 bits"))?;
    if buffer_ptr & 1 != 0 {
        return Err(PostflightError::new(
            "KECCAKF buffer pointer must be two-byte aligned",
        ));
    }
    let domain_end = if pointer_max_bits < 32 {
        1u64 << pointer_max_bits
    } else {
        1u64 << 32
    };
    if u64::from(buffer_ptr) + KECCAK_WIDTH_BYTES as u64 > domain_end {
        return Err(PostflightError::new(
            "KECCAKF state exceeds the pointer domain",
        ));
    }

    let mut preimage_buffer_bytes = [0u8; KECCAK_WIDTH_BYTES];
    for word_index in 0..KECCAK_WIDTH_MEM_OPS {
        let pointer = byte_ptr_to_u16_ptr_value(buffer_ptr) + (word_index * BLOCK_FE_WIDTH) as u32;
        let previous = replay.peek_u16(MEMORY_AS, pointer)?;
        preimage_buffer_bytes
            [word_index * MEMORY_BLOCK_BYTES..(word_index + 1) * MEMORY_BLOCK_BYTES]
            .copy_from_slice(&u16_block_to_bytes(previous));
    }
    let postimage = keccakf_postimage_bytes(&preimage_buffer_bytes);
    let mut buffer_prev_timestamps = [0; KECCAK_WIDTH_MEM_OPS];
    for (word_index, bytes) in postimage.chunks_exact(MEMORY_BLOCK_BYTES).enumerate() {
        let pointer = byte_ptr_to_u16_ptr_value(buffer_ptr) + (word_index * BLOCK_FE_WIDTH) as u32;
        let access = replay.write_u16(
            MEMORY_AS,
            pointer,
            bytes_to_u16_block(bytes.try_into().expect("chunk length is fixed")),
        )?;
        buffer_prev_timestamps[word_index] = access.previous_timestamp;
        let previous_bytes = u16_block_to_bytes(access.previous_value);
        if previous_bytes.as_slice()
            != &preimage_buffer_bytes
                [word_index * MEMORY_BLOCK_BYTES..(word_index + 1) * MEMORY_BLOCK_BYTES]
        {
            return Err(PostflightError::new(
                "KECCAKF peek did not resolve the write predecessor",
            ));
        }
    }
    let next_pc = pc
        .checked_add(DEFAULT_PC_STEP)
        .ok_or_else(|| PostflightError::new("KECCAKF program counter overflow"))?;
    replay.finish(next_pc)?;

    Ok(KeccakfReplay {
        pc,
        timestamp,
        rd_ptr,
        buffer_ptr,
        rd_prev_timestamp: rd.previous_timestamp,
        buffer_prev_timestamps,
        preimage_buffer_bytes,
        postimage_buffer_bytes: postimage,
    })
}
