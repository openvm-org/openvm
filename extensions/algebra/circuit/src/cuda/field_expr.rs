use std::{mem::size_of, sync::Arc};

use num_bigint::BigUint;
use openvm_circuit::arch::{
    rvr::cuda::{GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript},
    VmChipWrapper,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_mod_circuit_builder::{device_program::serialize_field_expr, FieldExpressionFiller};
use openvm_riscv_adapters::{Rv64VecHeapAdapterCols, Rv64VecHeapAdapterFiller};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use super::{
    cuda_abi,
    vec_heap::{
        checked_trace_shape, gather_vec_heap_trace_inputs, gather_vec_heap_trace_inputs_device,
        generate_field_expression_ctx_from_projection, DeviceVecHeapProjection,
    },
    DeferredGpuRangeCheckerCounts,
};
use crate::fields::get_field_type;

const MAX_FIELD_EXPR_SCRATCH_BYTES: usize = 128 << 20;
const MAX_FIELD_EXPR_LOCAL_BYTES: usize = 32 << 20;
const MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD: usize = 512;

fn supports_device_modulus(modulus: &BigUint) -> bool {
    get_field_type(modulus).is_some()
}

pub struct FieldExprReplayChip<const NUM_READS: usize, const BLOCKS: usize> {
    mode: FieldExprReplayMode<NUM_READS, BLOCKS>,
}

enum FieldExprReplayMode<const NUM_READS: usize, const BLOCKS: usize> {
    Gpu(FieldExprReplayChipGpu<NUM_READS, BLOCKS>),
    Cpu {
        device_ctx: GpuDeviceCtx,
        local_opcodes: Vec<usize>,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
    },
}

impl<const NUM_READS: usize, const BLOCKS: usize> FieldExprReplayChip<NUM_READS, BLOCKS> {
    pub fn new(
        chip: &VmChipWrapper<
            F,
            FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>,
        >,
        opcode_base: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Self, GpuRvrInputError> {
        if range_checker.count.len() != chip.inner.range_checker.count.len() {
            return Err(GpuRvrInputError::InvalidTranscript(
                "field-expression range-count shape mismatch".to_string(),
            ));
        }
        let pointer_max_bits = chip.inner.adapter().pointer_max_bits();
        let timestamp_max_bits = chip.mem_helper.timestamp_max_bits();
        let modulus = chip.inner.expr.program().prime();
        // The device interpreter uses Fermat inversion. Only the field implementations in this
        // crate are trusted prime moduli; all other moduli preserve the CPU executor's extended-
        // GCD division semantics.
        if !supports_device_modulus(modulus) {
            if !range_checker
                .cpu_chip
                .as_ref()
                .is_some_and(|cpu_chip| Arc::ptr_eq(cpu_chip, &chip.inner.range_checker))
            {
                return Err(GpuRvrInputError::InvalidTranscript(
                    "field-expression CPU fallback range checker is not hybrid-wired".to_string(),
                ));
            }
            return Ok(Self {
                mode: FieldExprReplayMode::Cpu {
                    device_ctx: range_checker.device_ctx.clone(),
                    local_opcodes: chip.inner.local_opcode_idx.clone(),
                    opcode_base,
                    pointer_max_bits,
                    timestamp_max_bits,
                },
            });
        }
        let serialized = serialize_field_expr(&chip.inner).map_err(|error| {
            GpuRvrInputError::InvalidTranscript(format!(
                "unsupported device field expression: {error:?}"
            ))
        })?;
        Ok(Self {
            mode: FieldExprReplayMode::Gpu(FieldExprReplayChipGpu::from_serialized(
                chip,
                serialized,
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
                range_checker,
            )?),
        })
    }

    pub fn opcode_base(&self) -> usize {
        match &self.mode {
            FieldExprReplayMode::Gpu(replay) => replay.opcode_base(),
            FieldExprReplayMode::Cpu { opcode_base, .. } => *opcode_base,
        }
    }

    pub fn local_opcodes(&self) -> &[usize] {
        match &self.mode {
            FieldExprReplayMode::Gpu(replay) => replay.local_opcodes(),
            FieldExprReplayMode::Cpu { local_opcodes, .. } => local_opcodes,
        }
    }

    pub fn generate_proving_ctx(
        &self,
        chip: &VmChipWrapper<
            F,
            FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>,
        >,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        match &self.mode {
            FieldExprReplayMode::Gpu(replay) => {
                replay.generate_proving_ctx(program, transcript, replay_plan)
            }
            FieldExprReplayMode::Cpu {
                device_ctx,
                local_opcodes,
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
            } => {
                let projection = gather_vec_heap_trace_inputs::<NUM_READS, BLOCKS>(
                    program,
                    transcript,
                    replay_plan,
                    *opcode_base,
                    local_opcodes,
                    *pointer_max_bits,
                    device_ctx,
                )?;
                generate_field_expression_ctx_from_projection(
                    chip,
                    projection,
                    *timestamp_max_bits,
                    device_ctx,
                )
            }
        }
    }
}

struct FieldExprReplayChipGpu<const NUM_READS: usize, const BLOCKS: usize> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    program: DeviceBuffer<u32>,
    local_opcodes: Vec<usize>,
    opcode_base: usize,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
    width: usize,
    aux_words_per_thread: usize,
}

impl<const NUM_READS: usize, const BLOCKS: usize> FieldExprReplayChipGpu<NUM_READS, BLOCKS> {
    fn from_serialized(
        chip: &VmChipWrapper<
            F,
            FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>,
        >,
        serialized: openvm_mod_circuit_builder::device_program::SerializedFieldExpr,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Self, GpuRvrInputError> {
        let num_input_bytes = chip
            .inner
            .num_inputs()
            .checked_mul(chip.inner.expr.program().canonical_num_limbs())
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression input width overflow".to_string(),
                )
            })?;
        let expected_input_bytes = NUM_READS
            .checked_mul(BLOCKS)
            .and_then(|blocks| blocks.checked_mul(openvm_circuit::arch::MEMORY_BLOCK_BYTES))
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript("VecHeap input width overflow".to_string())
            })?;
        if num_input_bytes != expected_input_bytes {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "field-expression input width {num_input_bytes} does not match VecHeap width {expected_input_bytes}"
            )));
        }
        let num_output_bytes = chip
            .inner
            .expr
            .program()
            .output_indices()
            .len()
            .checked_mul(chip.inner.expr.program().canonical_num_limbs())
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression output width overflow".to_string(),
                )
            })?;
        let expected_output_bytes = BLOCKS
            .checked_mul(openvm_circuit::arch::MEMORY_BLOCK_BYTES)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript("VecHeap output width overflow".to_string())
            })?;
        if num_output_bytes != expected_output_bytes {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "field-expression output width {num_output_bytes} does not match VecHeap width {expected_output_bytes}"
            )));
        }
        let adapter_width = Rv64VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS>::width();
        let core_width = BaseAir::<F>::width(&chip.inner.expr);
        if serialized.core_width != core_width {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "serialized field-expression width {} does not match AIR width {core_width}",
                serialized.core_width
            )));
        }
        let width = adapter_width.checked_add(core_width).ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript("field-expression trace width overflow".to_string())
        })?;
        let program = serialized
            .blob
            .as_slice()
            .to_device_on(&range_checker.device_ctx)?;
        Ok(Self {
            range_checker,
            program,
            local_opcodes: chip.inner.local_opcode_idx.clone(),
            opcode_base,
            pointer_max_bits,
            timestamp_max_bits,
            width,
            aux_words_per_thread: serialized.aux_words_per_thread,
        })
    }

    pub fn opcode_base(&self) -> usize {
        self.opcode_base
    }

    pub fn local_opcodes(&self) -> &[usize] {
        &self.local_opcodes
    }

    pub fn generate_proving_ctx(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        let projection = gather_vec_heap_trace_inputs_device::<NUM_READS, BLOCKS>(
            program,
            transcript,
            replay_plan,
            self.opcode_base,
            &self.local_opcodes,
            self.pointer_max_bits,
            device_ctx,
        )?;
        if projection.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        self.generate_from_projection(transcript, projection)
    }

    fn generate_from_projection(
        &self,
        transcript: &GpuRvrTranscript,
        projection: DeviceVecHeapProjection<NUM_READS, BLOCKS>,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        let pointer_max_bits = u32::try_from(self.pointer_max_bits).map_err(|_| {
            GpuRvrInputError::InvalidTranscript(
                "field-expression pointer width does not fit u32".to_string(),
            )
        })?;
        let timestamp_max_bits = u32::try_from(self.timestamp_max_bits).map_err(|_| {
            GpuRvrInputError::InvalidTranscript(
                "field-expression timestamp width does not fit u32".to_string(),
            )
        })?;
        let (height, _) =
            checked_trace_shape(projection.len(), self.width, self.timestamp_max_bits)?;
        let trace = DeviceMatrix::<F>::with_capacity_on(height, self.width, device_ctx);
        let delta = DeviceBuffer::with_capacity_on(self.range_checker.count.len(), device_ctx);
        delta.fill_zero_on(device_ctx)?;
        let max_scratch_words = MAX_FIELD_EXPR_SCRATCH_BYTES / size_of::<u32>();
        let launch = unsafe {
            cuda_abi::field_expr_replay_launch_config::<NUM_READS, BLOCKS>(
                height,
                self.aux_words_per_thread,
                max_scratch_words,
            )?
        };
        let expected_active_threads = launch
            .grid_blocks
            .checked_mul(launch.block_threads)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression launch thread count overflow".to_string(),
                )
            })?;
        let expected_scratch_words = expected_active_threads
            .checked_mul(self.aux_words_per_thread)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression launch scratch size overflow".to_string(),
                )
            })?;
        let scratch_bytes = launch
            .scratch_words
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression launch scratch byte size overflow".to_string(),
                )
            })?;
        let local_bytes = launch
            .active_threads
            .checked_mul(launch.local_bytes_per_thread)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "field-expression launch local-memory size overflow".to_string(),
                )
            })?;
        if launch.grid_blocks == 0
            || launch.block_threads == 0
            || launch.active_threads != expected_active_threads
            || launch.scratch_words == 0
            || launch.scratch_words != expected_scratch_words
            || launch.scratch_words > max_scratch_words
            || scratch_bytes > MAX_FIELD_EXPR_SCRATCH_BYTES
            || launch.local_bytes_per_thread > MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD
            || local_bytes > MAX_FIELD_EXPR_LOCAL_BYTES
        {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "invalid field-expression launch: grid={}, block={}, active={}, scratch={} words/{} bytes, local={} bytes/thread/{} bytes total",
                launch.grid_blocks,
                launch.block_threads,
                launch.active_threads,
                launch.scratch_words,
                scratch_bytes,
                launch.local_bytes_per_thread,
                local_bytes,
            )));
        }
        let scratch = DeviceBuffer::<u32>::with_capacity_on(launch.scratch_words, device_ctx);
        unsafe {
            cuda_abi::field_expr_replay_tracegen(
                trace.buffer(),
                height,
                &projection.inputs,
                &self.program,
                &delta,
                &scratch,
                self.aux_words_per_thread,
                launch,
                pointer_max_bits,
                timestamp_max_bits,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        transcript.synchronize()?;
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "field-expression replay rejected transcript with code {error:#010x}"
            )));
        }
        drop(scratch);
        drop(projection);
        DeferredGpuRangeCheckerCounts {
            target: self.range_checker.count.clone(),
            delta,
            device_ctx: device_ctx.clone(),
        }
        .commit()?;
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use openvm_circuit_primitives::bigint::utils::secp256k1_coord_prime;

    use super::supports_device_modulus;

    #[test]
    fn device_modulus_gate_is_deterministic_and_conservative() {
        assert!(supports_device_modulus(&secp256k1_coord_prime()));
        assert!(!supports_device_modulus(&BigUint::from(15u32)));
        assert!(!supports_device_modulus(&BigUint::from(16u32)));
    }
}
