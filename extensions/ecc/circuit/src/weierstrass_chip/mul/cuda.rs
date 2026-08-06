//! GPU projection for the `EC_MUL` chip.
//!
//! The shared vec-heap gather cannot describe this opcode: its two heap reads have different block
//! counts, and its trace is [`EC_MUL_TOTAL_ROWS`] rows per instruction rather than one. A dedicated
//! kernel gathers the same fields the host postflight projects, and the resulting projections feed
//! [`build_ec_mul_trace`] — the identical row encoding the CPU prover uses.

use openvm_algebra_circuit::cuda::gather_ec_mul;
use openvm_circuit::arch::cuda::postflight::{
    GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
};
use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    riscv::{MEMORY_AS, REGISTER_AS},
    VmOpcode,
};
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;

use super::{build_ec_mul_trace, EcMulChip, EcMulTraceInput};

/// Gathers every `EC_MUL` and `SETUP_EC_MUL` projection for one curve, then builds the trace.
///
/// Instructions are gathered per opcode into one device buffer, copied back, and sorted by
/// timestamp so the row order matches the host postflight's, which sorts the same way.
pub(crate) fn generate_ec_mul_trace_from_gpu_postflight<
    const NUM_LIMBS: usize,
    const BLOCKS: usize,
>(
    chip: &EcMulChip<F, NUM_LIMBS, BLOCKS>,
    opcode_base: usize,
    program: &GpuPostflightProgram,
    transcript: &GpuPostflightTranscript,
    replay_plan: &GpuPostflightPlan,
    device_ctx: &GpuDeviceCtx,
) -> Result<RowMajorMatrix<F>, GpuPostflightError> {
    program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;

    let mut ranges = Vec::with_capacity(2);
    let mut num_rows = 0usize;
    for (local, is_setup) in [
        (WeierstrassOpcode::EC_MUL, false),
        (WeierstrassOpcode::SETUP_EC_MUL, true),
    ] {
        let opcode = opcode_base
            .checked_add(local as usize)
            .ok_or(GpuPostflightError::OpcodeTooLarge(opcode_base))?;
        let opcode =
            u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))?;
        let range = replay_plan.opcode_range(VmOpcode::from_usize(opcode as usize));
        num_rows = num_rows.checked_add(range.len()).ok_or_else(|| {
            GpuPostflightError::InvalidTranscript("EC_MUL projection length overflow".to_string())
        })?;
        ranges.push((opcode, is_setup, range));
    }

    // An unused chip yields no rows. `build_ec_mul_trace` returns a zero-height matrix, which the
    // caller must not upload, so this is reported to the caller rather than handled here.
    let projection: DeviceBuffer<EcMulTraceInput<BLOCKS>> = if num_rows == 0 {
        DeviceBuffer::new()
    } else {
        DeviceBuffer::with_capacity_on(num_rows, device_ctx)
    };

    let pointer_max_bits = u32::try_from(chip.ptr_max_bits).map_err(|_| {
        GpuPostflightError::InvalidTranscript("EC_MUL pointer width does not fit u32".to_string())
    })?;
    let mut output_start = 0usize;
    for (opcode, is_setup, range) in ranges {
        if range.is_empty() {
            continue;
        }
        // SAFETY: `EcMulTraceInput` is `repr(C)` and its size is asserted equal to the kernel's
        // `EcMulTraceInput<BLOCKS>` on both sides; the device views all belong to `device_ctx`.
        unsafe {
            gather_ec_mul(
                &projection,
                output_start,
                BLOCKS,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                range.start,
                range.len(),
                opcode,
                is_setup,
                REGISTER_AS,
                MEMORY_AS,
                pointer_max_bits,
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
            "EC_MUL projection rejected transcript with code {error}"
        )));
    }

    let mut inputs = projection.to_host_on(device_ctx)?;
    // The host postflight sorts its steps by timestamp; matching that keeps both backends' row
    // order identical, which the digest row's execution-bus interaction depends on.
    inputs.sort_unstable_by_key(|input| input.start_timestamp());

    build_ec_mul_trace::<F, NUM_LIMBS, BLOCKS>(chip, &inputs).map_err(|error| {
        GpuPostflightError::InvalidTranscript(format!("EC_MUL trace generation failed: {error:?}"))
    })
}
