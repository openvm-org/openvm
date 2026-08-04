use std::{
    mem::size_of,
    sync::{atomic::Ordering, Arc},
};

use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        VmChipWrapper,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_primitives::{
    hybrid_chip::cpu_proving_ctx_to_gpu, var_range::VariableRangeCheckerChip,
};
use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    prelude::{F, SC},
    GpuBackend,
};
use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::FieldExpressionFiller;
use openvm_riscv_adapters::{
    vec_heap_u16_blocks_to_bytes, Rv64VecHeapAdapterCols, Rv64VecHeapAdapterFiller,
    VecHeapTraceInput,
};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*, prover::AirProvingContext,
};

use super::cuda_abi;

// Algebra AIRs have degree at most three. For BabyBear, the VM's proof-domain
// check therefore permits at most 2^(27 - ceil(log2(3 - 1))) rows.
const MAX_ALGEBRA_TRACE_HEIGHT: usize = 1 << 26;

pub(crate) fn checked_trace_shape(
    num_rows: usize,
    width: usize,
    timestamp_max_bits: usize,
) -> Result<(usize, usize), GpuPostflightError> {
    let timestamp_bits = u32::try_from(timestamp_max_bits).map_err(|_| {
        GpuPostflightError::InvalidConfiguration(
            "timestamp width cannot be represented as a host trace height".to_string(),
        )
    })?;
    let timestamp_row_limit = 1usize.checked_shl(timestamp_bits).ok_or_else(|| {
        GpuPostflightError::InvalidConfiguration(
            "timestamp width cannot be represented as a host trace height".to_string(),
        )
    })?;
    let max_height = timestamp_row_limit.min(MAX_ALGEBRA_TRACE_HEIGHT);
    let height = num_rows.checked_next_power_of_two().ok_or_else(|| {
        GpuPostflightError::InvalidTranscript("field-expression trace height overflow".to_string())
    })?;
    if height > max_height {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "field-expression trace height {height} exceeds maximum {max_height}"
        )));
    }
    let cells = height.checked_mul(width).ok_or_else(|| {
        GpuPostflightError::InvalidTranscript("field-expression trace size overflow".to_string())
    })?;
    Ok((height, cells))
}

/// Projects only the semantic values required by a vector-heap adapter and a
/// field-expression core. The device allocation has exactly one entry per
/// selected execution.
pub(crate) struct DeviceVecHeapProjection<const NUM_READS: usize, const BLOCKS: usize> {
    pub(crate) inputs: DeviceBuffer<VecHeapTraceInput<NUM_READS, BLOCKS>>,
}

impl<const NUM_READS: usize, const BLOCKS: usize> DeviceVecHeapProjection<NUM_READS, BLOCKS> {
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.len() == 0
    }
}

/// Gathers the bounded VecHeap replay projection without copying it back to the
/// host. Direct GPU trace generators consume this allocation and drop it before
/// proving starts.
pub(crate) fn gather_vec_heap_trace_inputs_device<const NUM_READS: usize, const BLOCKS: usize>(
    program: &GpuPostflightProgram,
    transcript: &GpuPostflightTranscript,
    replay_plan: &GpuPostflightPlan,
    opcode_base: usize,
    local_opcodes: &[usize],
    pointer_max_bits: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceVecHeapProjection<NUM_READS, BLOCKS>, GpuPostflightError> {
    program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
    if !matches!(
        (NUM_READS, BLOCKS),
        (2, 4) | (2, 6) | (2, 8) | (2, 12) | (1, 8) | (1, 12)
    ) {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "unsupported VecHeap replay shape ({NUM_READS}, {BLOCKS})"
        )));
    }
    if local_opcodes
        .iter()
        .enumerate()
        .any(|(i, opcode)| local_opcodes[..i].contains(opcode))
    {
        return Err(GpuPostflightError::InvalidTranscript(
            "duplicate VecHeap opcode ownership".to_string(),
        ));
    }

    let ranges = local_opcodes
        .iter()
        .map(|&local| {
            let opcode = opcode_base.checked_add(local).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript("VecHeap opcode overflow".to_string())
            })?;
            u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))?;
            u32::try_from(local).map_err(|_| GpuPostflightError::OpcodeTooLarge(local))?;
            Ok((
                local,
                opcode,
                replay_plan.opcode_range(VmOpcode::from_usize(opcode)),
            ))
        })
        .collect::<Result<Vec<_>, GpuPostflightError>>()?;
    let num_rows = ranges.iter().try_fold(0usize, |total, (_, _, range)| {
        total.checked_add(range.len()).ok_or_else(|| {
            GpuPostflightError::InvalidTranscript("VecHeap projection length overflow".to_string())
        })
    })?;
    if num_rows == 0 {
        return Ok(DeviceVecHeapProjection {
            inputs: DeviceBuffer::new(),
        });
    }
    if num_rows > MAX_ALGEBRA_TRACE_HEIGHT
        || num_rows
            .checked_mul(size_of::<VecHeapTraceInput<NUM_READS, BLOCKS>>())
            .is_none()
    {
        return Err(GpuPostflightError::InvalidTranscript(
            "VecHeap projection exceeds the algebra replay allocation limit".to_string(),
        ));
    }
    let projection = DeviceBuffer::with_capacity_on(num_rows, device_ctx);
    let mut output_start = 0;
    for (local_opcode, opcode, range) in ranges {
        if range.is_empty() {
            continue;
        }
        unsafe {
            cuda_abi::gather_vec_heap(
                &projection,
                output_start,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                range.start,
                range.len(),
                u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))?,
                u32::try_from(local_opcode)
                    .map_err(|_| GpuPostflightError::OpcodeTooLarge(local_opcode))?,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                u32::try_from(pointer_max_bits).map_err(|_| {
                    GpuPostflightError::InvalidTranscript(
                        "VecHeap pointer width does not fit u32".to_string(),
                    )
                })?,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        output_start += range.len();
    }
    debug_assert_eq!(output_start, num_rows);
    let error = transcript.error_code()?;
    if error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "VecHeap projection rejected transcript with code {error}"
        )));
    }
    Ok(DeviceVecHeapProjection { inputs: projection })
}

pub(crate) fn gather_vec_heap_trace_inputs<const NUM_READS: usize, const BLOCKS: usize>(
    program: &GpuPostflightProgram,
    transcript: &GpuPostflightTranscript,
    replay_plan: &GpuPostflightPlan,
    opcode_base: usize,
    local_opcodes: &[usize],
    pointer_max_bits: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<Vec<VecHeapTraceInput<NUM_READS, BLOCKS>>, GpuPostflightError> {
    gather_vec_heap_trace_inputs_device(
        program,
        transcript,
        replay_plan,
        opcode_base,
        local_opcodes,
        pointer_max_bits,
        device_ctx,
    )?
    .inputs
    .to_host_on(device_ctx)
    .map_err(Into::into)
}

fn flatten_blocks<const OUTER: usize, const BLOCKS: usize>(
    blocks: &[[[u16; 4]; BLOCKS]; OUTER],
) -> Vec<u8> {
    vec_heap_u16_blocks_to_bytes(blocks.iter().flatten().flatten())
}

fn flatten_writes<const BLOCKS: usize>(blocks: &[[u16; 4]; BLOCKS]) -> Vec<u8> {
    vec_heap_u16_blocks_to_bytes(blocks.iter().flatten())
}

/// Builds a CPU field-expression trace directly from the bounded semantic
/// projection, then transfers only the final trace to the GPU. Temporary range
/// counts are merged atomically only after every row has validated.
pub(crate) fn generate_field_expression_ctx_from_projection<
    const NUM_READS: usize,
    const BLOCKS: usize,
>(
    chip: &VmChipWrapper<
        F,
        FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>,
    >,
    projection: Vec<VecHeapTraceInput<NUM_READS, BLOCKS>>,
    timestamp_max_bits: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
    if projection.is_empty() {
        return Ok(AirProvingContext::simple_no_pis(
            openvm_cuda_backend::base::DeviceMatrix::dummy(),
        ));
    }
    let adapter_width = Rv64VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS>::width();
    let width = adapter_width
        .checked_add(BaseAir::<F>::width(&chip.inner.expr))
        .ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "field-expression trace width overflow".to_string(),
            )
        })?;
    let (height, cells) = checked_trace_shape(projection.len(), width, timestamp_max_bits)?;
    let mut values = F::zero_vec(cells);

    let temporary_range_checker = Arc::new(VariableRangeCheckerChip::new(
        chip.inner.range_checker.bus(),
    ));
    let temporary_mem_helper =
        SharedMemoryHelper::new(temporary_range_checker.clone(), timestamp_max_bits);
    let mem_helper = temporary_mem_helper.as_borrowed();
    projection
        .par_iter()
        .zip(values.par_chunks_exact_mut(width))
        .try_for_each(|(input, row)| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let read_bytes = flatten_blocks(&input.heap_reads);
            let write_bytes = flatten_writes(&input.writes);
            chip.inner
                .fill_trace_row_from_execution_data(
                    temporary_range_checker.as_ref(),
                    input.local_opcode as usize,
                    &read_bytes,
                    Some(&write_bytes),
                    core_row,
                )
                .map_err(|error| {
                    GpuPostflightError::InvalidTranscript(format!(
                        "field-expression projection failed validation: {error:?}"
                    ))
                })?;
            chip.inner.adapter().fill_trace_row_from_projection(
                temporary_range_checker.as_ref(),
                &mem_helper,
                adapter_row,
                input,
            );
            Ok::<(), GpuPostflightError>(())
        })?;
    if projection.len() < height {
        let mut dummy_row = F::zero_vec(width);
        chip.inner
            .fill_dummy_core_row(&mut dummy_row[adapter_width..]);
        values
            .par_chunks_exact_mut(width)
            .skip(projection.len())
            .for_each(|row| row.copy_from_slice(&dummy_row));
    }
    drop(projection);

    let cpu_ctx =
        AirProvingContext::<CpuBackend<SC>>::simple_no_pis(RowMajorMatrix::new(values, width));
    let gpu_ctx = cpu_proving_ctx_to_gpu(cpu_ctx, device_ctx);
    if chip.inner.range_checker.count.len() != temporary_range_checker.count.len() {
        return Err(GpuPostflightError::InvalidTranscript(
            "field-expression range-count shape mismatch".to_string(),
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
    Ok(gpu_ctx)
}
