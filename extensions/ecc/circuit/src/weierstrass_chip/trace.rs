use std::sync::{atomic::Ordering, Arc};

use openvm_circuit::{
    arch::{
        Postflight, PostflightError, PostflightStep, VmChipWrapper, BLOCK_FE_WIDTH,
        MEMORY_BLOCK_BYTES,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::FieldExpressionFiller;
use openvm_riscv_adapters::{
    vec_heap_u16_blocks_to_bytes, VecHeapAdapterCols, VecHeapAdapterFiller, VecHeapTraceInput,
};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};

use super::WeierstrassChip;

fn checked_u16_pointer(byte_pointer: u32, what: &str) -> Result<u32, PostflightError> {
    if byte_pointer & 1 != 0 {
        return Err(PostflightError::new(format!(
            "{what} byte pointer {byte_pointer:#x} is not u16-aligned"
        )));
    }
    Ok(byte_pointer >> 1)
}

fn pointer_from_register(
    value: [u16; BLOCK_FE_WIDTH],
    pointer_max_bits: usize,
) -> Result<u32, PostflightError> {
    if value[2] != 0 || value[3] != 0 {
        return Err(PostflightError::new(
            "vector-heap pointer register has nonzero upper 32 bits",
        ));
    }
    let pointer = u32::from(value[0]) | (u32::from(value[1]) << 16);
    if pointer_max_bits > u32::BITS as usize || u64::from(pointer) >= (1u64 << pointer_max_bits) {
        return Err(PostflightError::new(format!(
            "vector-heap pointer {pointer:#x} exceeds {pointer_max_bits}-bit address space"
        )));
    }
    Ok(pointer)
}

fn add_byte_offset(
    base: u32,
    block: usize,
    pointer_max_bits: usize,
) -> Result<u32, PostflightError> {
    let offset = block
        .checked_mul(MEMORY_BLOCK_BYTES)
        .and_then(|offset| u32::try_from(offset).ok())
        .ok_or_else(|| PostflightError::new("vector-heap block offset overflow"))?;
    let pointer = base
        .checked_add(offset)
        .ok_or_else(|| PostflightError::new("vector-heap byte pointer overflow"))?;
    if pointer_max_bits > u32::BITS as usize || u64::from(pointer) >= (1u64 << pointer_max_bits) {
        return Err(PostflightError::new(format!(
            "vector-heap byte pointer {pointer:#x} exceeds {pointer_max_bits}-bit address space"
        )));
    }
    Ok(pointer)
}

fn project_step<F: PrimeField32, const NUM_READS: usize, const BLOCKS: usize>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    local_opcode: usize,
    pointer_max_bits: usize,
) -> Result<VecHeapTraceInput<NUM_READS, BLOCKS>, PostflightError> {
    let instruction = postflight.instruction(step);
    if !matches!(NUM_READS, 1 | 2) {
        return Err(PostflightError::new(format!(
            "unsupported vector-heap read count {NUM_READS}"
        )));
    }
    if instruction.d.as_canonical_u32() != REGISTER_AS
        || instruction.e.as_canonical_u32() != MEMORY_AS
    {
        return Err(PostflightError::new(
            "vector-heap instruction uses invalid address spaces",
        ));
    }
    if (NUM_READS == 1 && instruction.c.as_canonical_u32() != 0)
        || instruction.f.as_canonical_u32() != 0
        || instruction.g.as_canonical_u32() != 0
    {
        return Err(PostflightError::new(
            "vector-heap instruction has nonzero unused operands",
        ));
    }

    let rs_ptrs = std::array::from_fn(|index| {
        if index == 0 {
            instruction.b.as_canonical_u32()
        } else {
            instruction.c.as_canonical_u32()
        }
    });
    let rd_ptr = instruction.a.as_canonical_u32();
    let mut replay = postflight.replay(step);

    let mut rs_vals = [0u32; NUM_READS];
    let mut rs_prev_timestamps = [0u32; NUM_READS];
    for index in 0..NUM_READS {
        let access = replay.read_u16(
            REGISTER_AS,
            checked_u16_pointer(rs_ptrs[index], "source register")?,
        )?;
        rs_vals[index] = pointer_from_register(access.value, pointer_max_bits)?;
        rs_prev_timestamps[index] = access.previous_timestamp;
    }
    let rd_access = replay.read_u16(
        REGISTER_AS,
        checked_u16_pointer(rd_ptr, "destination register")?,
    )?;
    let rd_val = pointer_from_register(rd_access.value, pointer_max_bits)?;

    let mut heap_prev_timestamps = [[0u32; BLOCKS]; NUM_READS];
    let mut heap_reads = [[[0u16; BLOCK_FE_WIDTH]; BLOCKS]; NUM_READS];
    for read in 0..NUM_READS {
        for block in 0..BLOCKS {
            let byte_pointer = add_byte_offset(rs_vals[read], block, pointer_max_bits)?;
            let access =
                replay.read_u16(MEMORY_AS, checked_u16_pointer(byte_pointer, "heap read")?)?;
            heap_reads[read][block] = access.value;
            heap_prev_timestamps[read][block] = access.previous_timestamp;
        }
    }

    let mut writes = [[0u16; BLOCK_FE_WIDTH]; BLOCKS];
    let mut write_predecessors = [[0u16; BLOCK_FE_WIDTH]; BLOCKS];
    let mut write_prev_timestamps = [0u32; BLOCKS];
    for block in 0..BLOCKS {
        let byte_pointer = add_byte_offset(rd_val, block, pointer_max_bits)?;
        let access = replay
            .write_observed_u16(MEMORY_AS, checked_u16_pointer(byte_pointer, "heap write")?)?;
        writes[block] = access.value;
        write_predecessors[block] = access.previous_value;
        write_prev_timestamps[block] = access.previous_timestamp;
    }

    let next_pc = postflight
        .pc(step)
        .checked_add(DEFAULT_PC_STEP)
        .ok_or_else(|| PostflightError::new("vector-heap next PC overflow"))?;
    replay.finish(next_pc)?;

    Ok(VecHeapTraceInput {
        from_pc: postflight.pc(step),
        from_timestamp: postflight.timestamp(step),
        local_opcode: u32::try_from(local_opcode)
            .map_err(|_| PostflightError::new("Weierstrass local opcode exceeds u32::MAX"))?,
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

fn generate_trace_from_postflights<
    F: PrimeField32 + Send + Sync + Clone,
    const NUM_READS: usize,
    const BLOCKS: usize,
>(
    chip: &VmChipWrapper<F, FieldExpressionFiller<VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>>,
    postflights: &[Postflight<'_, F>],
    opcode_base: usize,
    local_opcodes: &[WeierstrassOpcode],
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let pointer_max_bits = chip.inner.adapter().pointer_max_bits();
    let mut selected_steps = Vec::new();
    for (postflight_index, postflight) in postflights.iter().enumerate() {
        let mut selected = Vec::new();
        for &local_opcode in local_opcodes {
            let local_opcode = local_opcode as usize;
            let global_opcode = opcode_base
                .checked_add(local_opcode)
                .ok_or_else(|| PostflightError::new("Weierstrass opcode overflow"))?;
            selected.extend(
                postflight
                    .steps(VmOpcode::from_usize(global_opcode))
                    .iter()
                    .copied()
                    .map(|step| (postflight_index, step, local_opcode)),
            );
        }
        selected.sort_unstable_by_key(|&(_, step, _)| postflight.timestamp(step));
        selected_steps.extend(selected);
    }

    let adapter_width = VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS>::width();
    let width = adapter_width
        .checked_add(BaseAir::<F>::width(&chip.inner.expr))
        .ok_or_else(|| PostflightError::new("Weierstrass trace width overflow"))?;
    let height = if selected_steps.is_empty() {
        0
    } else {
        selected_steps
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| PostflightError::new("Weierstrass trace height overflow"))?
    };
    let cells = height
        .checked_mul(width)
        .ok_or_else(|| PostflightError::new("Weierstrass trace size overflow"))?;
    let mut trace = RowMajorMatrix::new(F::zero_vec(cells), width);

    let temporary_range_checker = Arc::new(VariableRangeCheckerChip::new(
        chip.inner.range_checker.bus(),
    ));
    let temporary_mem_helper = SharedMemoryHelper::new(
        temporary_range_checker.clone(),
        chip.mem_helper.timestamp_max_bits(),
    );
    let mem_helper = temporary_mem_helper.as_borrowed();
    trace.values[..selected_steps.len() * width]
        .par_chunks_exact_mut(width)
        .zip(selected_steps.par_iter().copied())
        .try_for_each(|(row, (postflight_index, step, local_opcode))| {
            let input = project_step::<F, NUM_READS, BLOCKS>(
                &postflights[postflight_index],
                step,
                local_opcode,
                pointer_max_bits,
            )?;
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let read_bytes =
                vec_heap_u16_blocks_to_bytes(input.heap_reads.iter().flatten().flatten());
            let write_bytes = vec_heap_u16_blocks_to_bytes(input.writes.iter().flatten());
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
                        "Weierstrass field-expression replay failed validation: {error:?}"
                    ))
                })?;
            chip.inner.adapter().fill_trace_row_from_projection(
                temporary_range_checker.as_ref(),
                &mem_helper,
                adapter_row,
                &input,
            );
            Ok::<_, PostflightError>(())
        })?;
    if selected_steps.len() < height {
        let mut dummy_row = F::zero_vec(width);
        chip.inner
            .fill_dummy_core_row(&mut dummy_row[adapter_width..]);
        trace.values[selected_steps.len() * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row| row.copy_from_slice(&dummy_row));
    }

    if chip.inner.range_checker.count.len() != temporary_range_checker.count.len() {
        return Err(PostflightError::new(
            "Weierstrass range-count shape mismatch",
        ));
    }
    for (destination, source) in chip
        .inner
        .range_checker
        .count
        .iter()
        .zip(&temporary_range_checker.count)
    {
        destination.fetch_add(source.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    Ok(trace)
}

pub(crate) fn generate_add_ne_trace_from_postflight<
    F: PrimeField32 + Send + Sync + Clone,
    const BLOCKS: usize,
>(
    chip: &WeierstrassChip<F, 2, BLOCKS>,
    postflight: &Postflight<'_, F>,
    opcode_base: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    generate_trace_from_postflights(
        chip,
        std::slice::from_ref(postflight),
        opcode_base,
        &[
            WeierstrassOpcode::EC_ADD_NE,
            WeierstrassOpcode::SETUP_EC_ADD_NE,
        ],
    )
}

#[cfg(test)]
pub(crate) fn generate_add_ne_trace_from_postflights<
    F: PrimeField32 + Send + Sync + Clone,
    const BLOCKS: usize,
>(
    chip: &WeierstrassChip<F, 2, BLOCKS>,
    postflights: &[Postflight<'_, F>],
    opcode_base: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    generate_trace_from_postflights(
        chip,
        postflights,
        opcode_base,
        &[
            WeierstrassOpcode::EC_ADD_NE,
            WeierstrassOpcode::SETUP_EC_ADD_NE,
        ],
    )
}

pub(crate) fn generate_double_trace_from_postflight<
    F: PrimeField32 + Send + Sync + Clone,
    const BLOCKS: usize,
>(
    chip: &WeierstrassChip<F, 1, BLOCKS>,
    postflight: &Postflight<'_, F>,
    opcode_base: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    generate_trace_from_postflights(
        chip,
        std::slice::from_ref(postflight),
        opcode_base,
        &[
            WeierstrassOpcode::EC_DOUBLE,
            WeierstrassOpcode::SETUP_EC_DOUBLE,
        ],
    )
}

#[cfg(test)]
pub(crate) fn generate_double_trace_from_postflights<
    F: PrimeField32 + Send + Sync + Clone,
    const BLOCKS: usize,
>(
    chip: &WeierstrassChip<F, 1, BLOCKS>,
    postflights: &[Postflight<'_, F>],
    opcode_base: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    generate_trace_from_postflights(
        chip,
        postflights,
        opcode_base,
        &[
            WeierstrassOpcode::EC_DOUBLE,
            WeierstrassOpcode::SETUP_EC_DOUBLE,
        ],
    )
}
