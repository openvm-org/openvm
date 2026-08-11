//! GPU trace generation for the `EC_MUL` chip.
//!
//! The shared vec-heap gather cannot describe this opcode: its two heap reads have different block
//! counts, and its trace is [`EC_MUL_TOTAL_ROWS`] rows per instruction rather than one. A dedicated
//! kernel gathers the same fields the host postflight projects, and [`EcMulTracegenGpu`] fills the
//! trace on the device. The host builder stays reachable through the postflight path, which is what
//! the device path is compared against.

use std::sync::Arc;

use openvm_algebra_circuit::cuda::{
    ec_mul_tracegen, gather_ec_mul, merge_range_counts, EcMulFillLaunchConfig,
};
use openvm_circuit::{
    arch::cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    riscv::{MEMORY_AS, REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::device_program::{
    serialize_field_expr_from_parts, DeviceOutputSource,
};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use super::{ec_mul_width, EcMulChip, EcMulTraceInput, EC_MUL_COMPUTE_ROWS, EC_MUL_TOTAL_ROWS};

/// Device memory the row-filling pass may use for per-thread scratch.
const MAX_EC_MUL_SCRATCH_BYTES: usize = 128 << 20;
const EC_MUL_FILL_BLOCK_THREADS: usize = 128;

/// Scratch the device's Jacobian point operations need, in slots of `u32_limbs` words.
///
/// Mirrors `EC_MUL_JACOBIAN_TEMPS` in `algebra/ec_mul_projective.cuh`. The device revalidates the
/// buffer this sizes before writing to it, so a drift here fails the launch rather than corrupting
/// memory.
const EC_MUL_JACOBIAN_TEMPS: usize = 14;
/// Mirrors `EC_MUL_MONT_WORKSPACE_WORDS`: the leading slots reserved for Montgomery multiplication.
const EC_MUL_MONT_WORKSPACE_WORDS: usize = 2;

/// Gathers every `EC_MUL` and `SETUP_EC_MUL` projection for one curve, in execution order.
///
/// The replay plan partitions steps by opcode, so the two opcodes arrive concatenated rather than
/// interleaved. Sorting is not required for soundness: every transition constraint is gated on
/// `in_instruction`, which is zero at an instruction boundary, and `scalar_acc` is reset by a local
/// constraint on `is_first_compute`, so each instruction is self-contained and their order is free.
/// It matches the host postflight, so both backends emit the same trace and one can check the
/// other.
fn gather_projections<const BLOCKS: usize>(
    ptr_max_bits: usize,
    opcode_base: usize,
    program: &GpuPostflightProgram,
    transcript: &GpuPostflightTranscript,
    replay_plan: &GpuPostflightPlan,
    device_ctx: &GpuDeviceCtx,
) -> Result<Vec<EcMulTraceInput<BLOCKS>>, GpuPostflightError> {
    program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;

    let mut ranges = Vec::with_capacity(2);
    let mut num_instructions = 0usize;
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
        num_instructions = num_instructions.checked_add(range.len()).ok_or_else(|| {
            GpuPostflightError::InvalidTranscript("EC_MUL projection length overflow".to_string())
        })?;
        ranges.push((opcode, is_setup, range));
    }
    if num_instructions == 0 {
        return Ok(Vec::new());
    }

    let projection: DeviceBuffer<EcMulTraceInput<BLOCKS>> =
        DeviceBuffer::with_capacity_on(num_instructions, device_ctx);
    let pointer_max_bits = u32::try_from(ptr_max_bits).map_err(|_| {
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
    debug_assert_eq!(output_start, num_instructions);

    let error = transcript.error_code()?;
    if error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "EC_MUL projection rejected transcript with code {error}"
        )));
    }

    let mut inputs = projection.to_host_on(device_ctx)?;
    inputs.sort_unstable_by_key(|input| input.start_timestamp());
    Ok(inputs)
}

/// Grid dimensions for the row-filling pass, sized so its scratch stays within budget.
fn fill_launch_config(
    height: usize,
    aux_words: usize,
    max_scratch_words: usize,
) -> Result<EcMulFillLaunchConfig, GpuPostflightError> {
    if aux_words == 0 {
        return Err(GpuPostflightError::InvalidConfiguration(
            "EC_MUL kernel requires nonzero scratch words per thread".to_string(),
        ));
    }
    let scratch_per_block = EC_MUL_FILL_BLOCK_THREADS
        .checked_mul(aux_words)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration("EC_MUL scratch overflow".to_string())
        })?;
    let scratch_limited = max_scratch_words / scratch_per_block;
    if scratch_limited == 0 {
        return Err(GpuPostflightError::ResourceLimitExceeded {
            resource: "EC_MUL scratch words per block",
            requested: scratch_per_block,
            limit: max_scratch_words,
        });
    }
    let grid_blocks = height
        .div_ceil(EC_MUL_FILL_BLOCK_THREADS)
        .min(scratch_limited)
        .max(1);
    Ok(EcMulFillLaunchConfig {
        grid_blocks,
        block_threads: EC_MUL_FILL_BLOCK_THREADS,
        scratch_words: grid_blocks * scratch_per_block,
    })
}

/// Device-side trace generation for one curve's `EC_MUL` chip.
///
/// The serialized expression is uploaded once, at construction, since it depends only on the curve.
pub(crate) struct EcMulTracegenGpu {
    program: DeviceBuffer<u32>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    aux_words: usize,
    expr_width: usize,
    u32_limbs: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
}

impl EcMulTracegenGpu {
    pub fn new<const NUM_LIMBS: usize, const BLOCKS: usize>(
        chip: &EcMulChip<F, NUM_LIMBS, BLOCKS>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Self, GpuPostflightError> {
        // The ladder drives its flags per row rather than per opcode, so the table declares only
        // the setup opcode. It exists to satisfy the device validator's shape rule; the kernels
        // never look an opcode up in it.
        let serialized = serialize_field_expr_from_parts(
            &chip.expr,
            &[WeierstrassOpcode::SETUP_EC_MUL as usize],
            &[],
            false,
            DeviceOutputSource::Computed,
        )
        .map_err(|error| {
            GpuPostflightError::InvalidConfiguration(format!(
                "unsupported EC_MUL device expression: {error:?}"
            ))
        })?;

        let expr_width = BaseAir::<F>::width(&chip.expr);
        if serialized.core_width != expr_width {
            return Err(GpuPostflightError::InvalidConfiguration(format!(
                "serialized EC_MUL width {} does not match AIR width {expr_width}",
                serialized.core_width
            )));
        }
        if range_checker.count.len() != chip.range_checker.count.len() {
            return Err(GpuPostflightError::InvalidConfiguration(
                "EC_MUL range-count shape mismatch".to_string(),
            ));
        }

        let device_ctx = range_checker.device_ctx.clone();
        let program = serialized.blob.as_slice().to_device_on(&device_ctx)?;
        Ok(Self {
            program,
            aux_words: serialized.aux_words_per_thread,
            expr_width,
            // Canonical limbs are bytes, so a `u32` limb spans four of them.
            u32_limbs: NUM_LIMBS.div_ceil(4),
            pointer_max_bits: u32::try_from(chip.ptr_max_bits).map_err(|_| {
                GpuPostflightError::InvalidConfiguration(
                    "EC_MUL pointer width does not fit u32".to_string(),
                )
            })?,
            timestamp_max_bits: u32::try_from(chip.mem_helper.timestamp_max_bits()).map_err(
                |_| {
                    GpuPostflightError::InvalidConfiguration(
                        "EC_MUL timestamp width does not fit u32".to_string(),
                    )
                },
            )?,
            range_checker,
        })
    }

    pub fn generate_proving_ctx<const NUM_LIMBS: usize, const BLOCKS: usize>(
        &self,
        chip: &EcMulChip<F, NUM_LIMBS, BLOCKS>,
        opcode_base: usize,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        let inputs = gather_projections::<BLOCKS>(
            chip.ptr_max_bits,
            opcode_base,
            program,
            transcript,
            replay_plan,
            device_ctx,
        )?;
        // Every configured curve registers a chip whether or not the program uses one. An unused
        // chip has no rows, and a zero-capacity device allocation is rejected.
        if inputs.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let num_instructions = inputs.len();
        let width = ec_mul_width::<NUM_LIMBS, BLOCKS>(self.expr_width);
        let height = next_power_of_two_or_zero(num_instructions * EC_MUL_TOTAL_ROWS);
        let max_scratch_words = MAX_EC_MUL_SCRATCH_BYTES / size_of::<u32>();
        let launch = fill_launch_config(height, self.aux_words, max_scratch_words)?;
        // The per-instruction pass indexes the same scratch by instruction rather than by row.
        let scratch_words = launch
            .scratch_words
            .max(num_instructions.saturating_mul(self.aux_words));
        if scratch_words > max_scratch_words {
            return Err(GpuPostflightError::ResourceLimitExceeded {
                resource: "EC_MUL scratch words",
                requested: scratch_words,
                limit: max_scratch_words,
            });
        }

        let projection = inputs.as_slice().to_device_on(device_ctx)?;
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        // The accumulator entering each compute row, as the affine bytes pass 2 reads as input.
        let affine = DeviceBuffer::<u8>::with_capacity_on(
            num_instructions * EC_MUL_COMPUTE_ROWS * 2 * NUM_LIMBS,
            device_ctx,
        );
        // Per instruction: Jacobian accumulators, the prefix products the batch inversion consumes,
        // then that thread's Montgomery workspace. Pass 1 only. Kept off the kernel's stack, where
        // it would cost occupancy for nothing.
        let ladder_slice_words = EC_MUL_COMPUTE_ROWS * 4 * self.u32_limbs
            + EC_MUL_MONT_WORKSPACE_WORDS * self.u32_limbs
            + 2
            + EC_MUL_JACOBIAN_TEMPS * self.u32_limbs;
        let ladder = DeviceBuffer::<u32>::with_capacity_on(
            num_instructions * ladder_slice_words,
            device_ctx,
        );
        let dummy_expr = DeviceBuffer::<F>::with_capacity_on(self.expr_width, device_ctx);
        let scratch = DeviceBuffer::<u32>::with_capacity_on(scratch_words, device_ctx);
        let delta = DeviceBuffer::<F>::with_capacity_on(self.range_checker.count.len(), device_ctx);
        delta.fill_zero_on(device_ctx)?;
        // The dummy row's own range checks are discarded: the AIR emits none when `is_valid` is
        // zero.
        let discarded =
            DeviceBuffer::<F>::with_capacity_on(self.range_checker.count.len(), device_ctx);
        discarded.fill_zero_on(device_ctx)?;

        // SAFETY: `EcMulTraceInput` matches the kernel's layout, as asserted on both sides, and
        // every buffer was allocated on `device_ctx`.
        unsafe {
            ec_mul_tracegen(
                trace.buffer(),
                height,
                width,
                NUM_LIMBS,
                BLOCKS,
                &projection,
                &self.program,
                &affine,
                &ladder,
                &dummy_expr,
                &delta,
                &discarded,
                &scratch,
                self.aux_words,
                launch,
                self.pointer_max_bits,
                self.timestamp_max_bits,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "EC_MUL trace generation rejected transcript with code {error:#010x}"
            )));
        }

        // Dropped after their kernels are enqueued on the owning stream, before proving starts.
        drop(discarded);
        drop(scratch);
        drop(ladder);
        drop(affine);
        drop(dummy_expr);
        drop(projection);

        // SAFETY: both histograms have the same length, checked at construction.
        unsafe {
            merge_range_counts(
                self.range_checker.count.as_ref(),
                &delta,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
