use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::{atomic::Ordering, Arc},
};

use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::{
    arch::{
        fill_trace_rows, Postflight, PostflightError, PostflightStep, U16Access, VmChipWrapper,
        BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE,
    },
    system::memory::SharedMemoryHelper,
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::FieldExpressionFiller;
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterCols, Rv64VecHeapAdapterCols, Rv64VecHeapAdapterFiller,
    VecHeapTraceInput,
};
use openvm_riscv_circuit::adapters::{
    byte_ptr_limbs_to_cell_ptr_limbs_value, cell_ptr_hi_bits, compute_block_add_carries,
    ptr_to_field_u16_limbs, u32_to_ptr_limbs, U16_BITS,
};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};

use crate::modular_chip::{ModularIsEqualCoreCols, ModularIsEqualU16Chip};

fn checked_u16_pointer(pointer: u32, description: &str) -> Result<u32, PostflightError> {
    if pointer & 1 != 0 {
        return Err(PostflightError::new(format!(
            "{description} byte pointer is not u16-aligned"
        )));
    }
    Ok(pointer >> 1)
}

fn checked_heap_pointer(
    value: [u16; BLOCK_FE_WIDTH],
    blocks: usize,
    pointer_max_bits: usize,
) -> Result<u32, PostflightError> {
    if value[2..].iter().any(|&limb| limb != 0) {
        return Err(PostflightError::new(
            "vector-heap pointer has nonzero upper 32 bits",
        ));
    }
    if pointer_max_bits > u32::BITS as usize {
        return Err(PostflightError::new(
            "vector-heap pointer width exceeds 32 bits",
        ));
    }
    let pointer = u32::from(value[0]) | (u32::from(value[1]) << U16_BITS);
    let last_byte = u64::from(pointer)
        .checked_add(
            u64::try_from(blocks)
                .ok()
                .and_then(|blocks| blocks.checked_mul(MEMORY_BLOCK_BYTES as u64))
                .and_then(|bytes| bytes.checked_sub(1))
                .ok_or_else(|| PostflightError::new("vector-heap access width overflow"))?,
        )
        .ok_or_else(|| PostflightError::new("vector-heap pointer overflow"))?;
    let pointer_limit = 1u64 << pointer_max_bits;
    if last_byte >= pointer_limit {
        return Err(PostflightError::new(
            "vector-heap access exceeds the configured pointer domain",
        ));
    }
    Ok(pointer)
}

fn merge_range_counts(
    destination: &VariableRangeCheckerChip,
    source: &VariableRangeCheckerChip,
) -> Result<(), PostflightError> {
    if destination.count.len() != source.count.len() {
        return Err(PostflightError::new("range-checker shape mismatch"));
    }
    for (destination, source) in destination.count.iter().zip(&source.count) {
        destination.fetch_add(source.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    Ok(())
}

fn write_u16_bytes<'a>(output: &mut Vec<u8>, limbs: impl IntoIterator<Item = &'a u16>) {
    output.clear();
    for limb in limbs {
        output.extend_from_slice(&limb.to_le_bytes());
    }
}

fn replay_vec_heap<const NUM_READS: usize, const BLOCKS: usize, F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    local_opcode: usize,
    pointer_max_bits: usize,
) -> Result<VecHeapTraceInput<NUM_READS, BLOCKS>, PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
        || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
    {
        return Err(PostflightError::new(
            "vector-heap instruction has invalid address spaces",
        ));
    }

    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let rs_ptrs = from_fn(|index| {
        if index == 0 {
            instruction.b.as_canonical_u32()
        } else {
            instruction.c.as_canonical_u32()
        }
    });
    let rd_ptr = instruction.a.as_canonical_u32();
    let mut rs_u16_ptrs = [0; NUM_READS];
    for index in 0..NUM_READS {
        rs_u16_ptrs[index] = checked_u16_pointer(rs_ptrs[index], "source register")?;
    }
    let rd_u16_ptr = checked_u16_pointer(rd_ptr, "destination register")?;

    let mut replay = postflight.replay(step);
    let mut rs_accesses: [Option<U16Access>; NUM_READS] = from_fn(|_| None);
    for index in 0..NUM_READS {
        rs_accesses[index] = Some(replay.read_u16(RV64_REGISTER_AS, rs_u16_ptrs[index])?);
    }
    let rd_access = replay.read_u16(RV64_REGISTER_AS, rd_u16_ptr)?;

    let mut rs_vals = [0; NUM_READS];
    let mut rs_prev_timestamps = [0; NUM_READS];
    let mut heap_prev_timestamps = [[0; BLOCKS]; NUM_READS];
    let mut heap_reads = [[[0; BLOCK_FE_WIDTH]; BLOCKS]; NUM_READS];
    for read_index in 0..NUM_READS {
        let access = rs_accesses[read_index]
            .take()
            .expect("every source register access is present");
        let pointer = checked_heap_pointer(access.value, BLOCKS, pointer_max_bits)?;
        rs_vals[read_index] = pointer;
        rs_prev_timestamps[read_index] = access.previous_timestamp;
        for block in 0..BLOCKS {
            let byte_pointer = pointer
                .checked_add((block * MEMORY_BLOCK_BYTES) as u32)
                .ok_or_else(|| PostflightError::new("vector-heap read pointer overflow"))?;
            let access = replay.read_u16(
                RV64_MEMORY_AS,
                checked_u16_pointer(byte_pointer, "heap read")?,
            )?;
            heap_prev_timestamps[read_index][block] = access.previous_timestamp;
            heap_reads[read_index][block] = access.value;
        }
    }

    let rd_val = checked_heap_pointer(rd_access.value, BLOCKS, pointer_max_bits)?;
    let mut write_prev_timestamps = [0; BLOCKS];
    let mut writes = [[0; BLOCK_FE_WIDTH]; BLOCKS];
    let mut write_predecessors = [[0; BLOCK_FE_WIDTH]; BLOCKS];
    for block in 0..BLOCKS {
        let byte_pointer = rd_val
            .checked_add((block * MEMORY_BLOCK_BYTES) as u32)
            .ok_or_else(|| PostflightError::new("vector-heap write pointer overflow"))?;
        let access = replay.write_observed_u16(
            RV64_MEMORY_AS,
            checked_u16_pointer(byte_pointer, "heap write")?,
        )?;
        write_prev_timestamps[block] = access.previous_timestamp;
        writes[block] = access.value;
        write_predecessors[block] = access.previous_value;
    }
    replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

    Ok(VecHeapTraceInput {
        from_pc,
        from_timestamp,
        local_opcode: u32::try_from(local_opcode)
            .map_err(|_| PostflightError::new("local opcode exceeds u32::MAX"))?,
        rs_ptrs,
        rd_ptr,
        rs_vals,
        rd_val,
        rs_prev_timestamps,
        rd_prev_timestamp: rd_access.previous_timestamp,
        heap_prev_timestamps,
        write_prev_timestamps,
        heap_reads,
        writes,
        write_predecessors,
    })
}

/// Generates a modular or Fp2 field-expression trace from immutable preflight history.
pub(crate) fn generate_field_expression_trace_from_postflight<
    F: PrimeField32 + Send + Sync + Clone,
    const BLOCKS: usize,
>(
    chip: &VmChipWrapper<F, FieldExpressionFiller<Rv64VecHeapAdapterFiller<2, BLOCKS, BLOCKS>>>,
    postflight: &Postflight<'_, F>,
    opcode_base: usize,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let rows_used =
        chip.inner
            .local_opcode_idx
            .iter()
            .try_fold(0usize, |rows, &local_opcode| {
                let opcode = opcode_base
                    .checked_add(local_opcode)
                    .ok_or_else(|| PostflightError::new("field-expression opcode overflow"))?;
                rows.checked_add(postflight.steps(VmOpcode::from_usize(opcode)).len())
                    .ok_or_else(|| PostflightError::new("field-expression trace height overflow"))
            })?;
    let adapter_width = Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width();
    let width = adapter_width
        .checked_add(BaseAir::<F>::width(&chip.inner.expr))
        .ok_or_else(|| PostflightError::new("field-expression trace width overflow"))?;
    let height = next_power_of_two_or_zero(rows_used);
    let cells = height
        .checked_mul(width)
        .ok_or_else(|| PostflightError::new("field-expression trace size overflow"))?;
    let mut trace = RowMajorMatrix::new(F::zero_vec(cells), width);

    let temporary_range_checker = Arc::new(VariableRangeCheckerChip::new(
        chip.inner.range_checker.bus(),
    ));
    let temporary_memory_helper = SharedMemoryHelper::new(
        temporary_range_checker.clone(),
        chip.mem_helper.timestamp_max_bits(),
    );
    let memory_helper = temporary_memory_helper.as_borrowed();

    let mut row_index = 0;
    for &local_opcode in &chip.inner.local_opcode_idx {
        let opcode = opcode_base
            .checked_add(local_opcode)
            .ok_or_else(|| PostflightError::new("field-expression opcode overflow"))?;
        let steps = postflight.steps(VmOpcode::from_usize(opcode));
        let rows_end = row_index + steps.len();
        trace.values[row_index * width..rows_end * width]
            .par_chunks_exact_mut(width)
            .zip(steps.par_iter())
            .try_for_each(|(row, &step)| {
                let input = replay_vec_heap::<2, BLOCKS, F>(
                    postflight,
                    step,
                    local_opcode,
                    pointer_max_bits,
                )?;
                let (adapter_row, core_row) = row.split_at_mut(adapter_width);
                let mut read_bytes = Vec::with_capacity(2 * BLOCKS * MEMORY_BLOCK_BYTES);
                let mut write_bytes = Vec::with_capacity(BLOCKS * MEMORY_BLOCK_BYTES);
                write_u16_bytes(&mut read_bytes, input.heap_reads.iter().flatten().flatten());
                write_u16_bytes(&mut write_bytes, input.writes.iter().flatten());
                chip.inner
                    .fill_trace_row_from_execution_data(
                        temporary_range_checker.as_ref(),
                        local_opcode,
                        &read_bytes,
                        Some(&write_bytes),
                        core_row,
                    )
                    .map_err(|error| {
                        PostflightError::new(format!(
                            "field-expression execution data is invalid: {error:?}"
                        ))
                    })?;
                chip.inner.adapter().fill_trace_row_from_projection(
                    temporary_range_checker.as_ref(),
                    &memory_helper,
                    adapter_row,
                    &input,
                );
                Ok::<(), PostflightError>(())
            })?;
        row_index = rows_end;
    }
    if row_index < height {
        let mut dummy_row = F::zero_vec(width);
        chip.inner
            .fill_dummy_core_row(&mut dummy_row[adapter_width..]);
        trace.values[row_index * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row| row.copy_from_slice(&dummy_row));
    }
    merge_range_counts(
        chip.inner.range_checker.as_ref(),
        temporary_range_checker.as_ref(),
    )?;
    Ok(trace)
}

/// Generates a modular equality trace from immutable preflight history.
pub(crate) fn generate_modular_is_equal_trace_from_postflight<
    F: PrimeField32,
    const NUM_LANES: usize,
    const TOTAL_LIMBS: usize,
>(
    chip: &ModularIsEqualU16Chip<F, TOTAL_LIMBS>,
    postflight: &Postflight<'_, F>,
    opcode_base: usize,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let local_opcodes = [
        Rv64ModularArithmeticOpcode::IS_EQ as usize,
        Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize,
    ];
    let rows_used = local_opcodes.iter().try_fold(0usize, |rows, &local| {
        let opcode = opcode_base
            .checked_add(local)
            .ok_or_else(|| PostflightError::new("modular equality opcode overflow"))?;
        rows.checked_add(postflight.steps(VmOpcode::from_usize(opcode)).len())
            .ok_or_else(|| PostflightError::new("modular equality trace height overflow"))
    })?;
    let adapter_width = Rv64IsEqualModU16AdapterCols::<F, 2, NUM_LANES>::width();
    let width = adapter_width
        .checked_add(ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width())
        .ok_or_else(|| PostflightError::new("modular equality trace width overflow"))?;
    let height = next_power_of_two_or_zero(rows_used);
    let cells = height
        .checked_mul(width)
        .ok_or_else(|| PostflightError::new("modular equality trace size overflow"))?;
    let mut trace = RowMajorMatrix::new(F::zero_vec(cells), width);

    let temporary_range_checker = Arc::new(VariableRangeCheckerChip::new(
        chip.inner.range_checker_chip.bus(),
    ));
    let temporary_memory_helper = SharedMemoryHelper::new(
        temporary_range_checker.clone(),
        chip.mem_helper.timestamp_max_bits(),
    );
    let memory_helper = temporary_memory_helper.as_borrowed();

    let mut row_index = 0;
    for local_opcode in local_opcodes {
        let opcode = opcode_base
            .checked_add(local_opcode)
            .ok_or_else(|| PostflightError::new("modular equality opcode overflow"))?;
        let steps = postflight.steps(VmOpcode::from_usize(opcode));
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let instruction = postflight.instruction(step);
            if instruction.a.as_canonical_u32() == 0
                || instruction.d.as_canonical_u32() != RV64_REGISTER_AS
                || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
            {
                return Err(PostflightError::new(
                    "modular equality instruction has invalid operands",
                ));
            }
            let from_pc = postflight.pc(step);
            let from_timestamp = postflight.timestamp(step);
            let rs_ptrs = [
                instruction.b.as_canonical_u32(),
                instruction.c.as_canonical_u32(),
            ];
            let rd_ptr = instruction.a.as_canonical_u32();
            let rs_u16_ptrs = [
                checked_u16_pointer(rs_ptrs[0], "source register")?,
                checked_u16_pointer(rs_ptrs[1], "source register")?,
            ];
            let rd_u16_ptr = checked_u16_pointer(rd_ptr, "destination register")?;

            let mut replay = postflight.replay(step);
            let rs_accesses = [
                replay.read_u16(RV64_REGISTER_AS, rs_u16_ptrs[0])?,
                replay.read_u16(RV64_REGISTER_AS, rs_u16_ptrs[1])?,
            ];
            let mut rs_vals = [0; 2];
            let mut heap_accesses: [[Option<U16Access>; NUM_LANES]; 2] =
                from_fn(|_| from_fn(|_| None));
            let mut inputs = [[0; TOTAL_LIMBS]; 2];
            for read in 0..2 {
                let pointer =
                    checked_heap_pointer(rs_accesses[read].value, NUM_LANES, pointer_max_bits)?;
                rs_vals[read] = pointer;
                for block in 0..NUM_LANES {
                    let byte_pointer = pointer
                        .checked_add((block * MEMORY_BLOCK_BYTES) as u32)
                        .ok_or_else(|| {
                            PostflightError::new("modular equality read pointer overflow")
                        })?;
                    let access = replay.read_u16(
                        RV64_MEMORY_AS,
                        checked_u16_pointer(byte_pointer, "heap read")?,
                    )?;
                    inputs[read][block * BLOCK_FE_WIDTH..(block + 1) * BLOCK_FE_WIDTH]
                        .copy_from_slice(&access.value);
                    heap_accesses[read][block] = Some(access);
                }
            }
            let mut write_value = [0; BLOCK_FE_WIDTH];
            write_value[0] = (inputs[0] == inputs[1]) as u16;
            let write = replay.write_u16(RV64_REGISTER_AS, rd_u16_ptr, write_value)?;
            replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let adapter_cols: &mut Rv64IsEqualModU16AdapterCols<F, 2, NUM_LANES> =
                adapter_row.borrow_mut();
            adapter_cols
                .writes_aux
                .set_prev_data(write.previous_value.map(F::from_u16));
            memory_helper.fill(
                write.previous_timestamp,
                write.timestamp,
                adapter_cols.writes_aux.as_mut(),
            );
            adapter_cols.rd_ptr = F::from_u32(rd_ptr);
            for read in 0..2 {
                for (block, access) in heap_accesses[read].iter_mut().enumerate() {
                    let access = access
                        .take()
                        .expect("every replayed heap access is present");
                    memory_helper.fill(
                        access.previous_timestamp,
                        access.timestamp,
                        adapter_cols.heap_read_aux[read][block].as_mut(),
                    );
                }
                memory_helper.fill(
                    rs_accesses[read].previous_timestamp,
                    rs_accesses[read].timestamp,
                    adapter_cols.rs_read_aux[read].as_mut(),
                );
                // Byte -> cell conversion carry plus one add-carry per heap block, with the
                // matching range-check counts (the AIR converts each base pointer once and adds
                // the per-block cell offset, all with multiplicity `is_valid`).
                let (conv_carry, base_cell) =
                    byte_ptr_limbs_to_cell_ptr_limbs_value(u32_to_ptr_limbs(rs_vals[read]));
                temporary_range_checker.add_count(base_cell[1], cell_ptr_hi_bits(pointer_max_bits));
                let add_carries = compute_block_add_carries(
                    &temporary_range_checker,
                    base_cell.map(|limb| limb as u16),
                    NUM_LANES,
                    (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32,
                );
                adapter_cols.rs_cell_carry[read] = F::from_u32(conv_carry);
                for (col, carry) in adapter_cols.reads_add_carry[read]
                    .iter_mut()
                    .zip(add_carries)
                {
                    *col = F::from_u32(carry);
                }
                adapter_cols.rs_val[read] = ptr_to_field_u16_limbs(rs_vals[read]);
                adapter_cols.rs_ptr[read] = F::from_u32(rs_ptrs[read]);
            }
            adapter_cols.from_state.timestamp = F::from_u32(from_timestamp);
            adapter_cols.from_state.pc = F::from_u32(from_pc);

            chip.inner.fill_trace_row_from_execution_data(
                temporary_range_checker.as_ref(),
                local_opcode == Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize,
                inputs[0],
                inputs[1],
                core_row,
            )?;
            Ok(())
        })?;
        row_index += steps.len();
    }
    merge_range_counts(
        chip.inner.range_checker_chip.as_ref(),
        temporary_range_checker.as_ref(),
    )?;
    Ok(trace)
}
