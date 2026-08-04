use std::{mem::size_of, sync::Arc};

use num_bigint::BigUint;
use openvm_circuit::arch::{
    cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    VmChipWrapper,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    common::set_device_by_id, copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx,
};
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

#[derive(Clone, Copy, Debug)]
struct ValidatedFieldExprKernelConfig(cuda_abi::FieldExprReplayKernelConfig);

fn resource_limit(resource: &'static str, requested: usize, limit: usize) -> GpuPostflightError {
    GpuPostflightError::ResourceLimitExceeded {
        resource,
        requested,
        limit,
    }
}

fn validate_kernel_config(
    kernel: cuda_abi::FieldExprReplayKernelConfig,
) -> Result<ValidatedFieldExprKernelConfig, GpuPostflightError> {
    if kernel.max_grid_blocks == 0 || kernel.block_threads == 0 || kernel.block_threads > 1024 {
        return Err(GpuPostflightError::InvalidConfiguration(format!(
            "field-expression kernel returned max_grid={} and block={}",
            kernel.max_grid_blocks, kernel.block_threads,
        )));
    }
    if kernel.local_bytes_per_thread > MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD {
        return Err(resource_limit(
            "field-expression local bytes per thread",
            kernel.local_bytes_per_thread,
            MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD,
        ));
    }
    Ok(ValidatedFieldExprKernelConfig(kernel))
}

fn validate_launch_config(
    launch: cuda_abi::FieldExprReplayLaunchConfig,
    aux_words_per_thread: usize,
    max_scratch_words: usize,
) -> Result<cuda_abi::FieldExprReplayLaunchConfig, GpuPostflightError> {
    let expected_active_threads = launch
        .grid_blocks
        .checked_mul(launch.block_threads)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch thread count overflow".to_string(),
            )
        })?;
    let expected_scratch_words = expected_active_threads
        .checked_mul(aux_words_per_thread)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch scratch size overflow".to_string(),
            )
        })?;
    let scratch_bytes = launch
        .scratch_words
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch scratch byte size overflow".to_string(),
            )
        })?;
    let local_bytes = launch
        .active_threads
        .checked_mul(launch.local_bytes_per_thread)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch local-memory size overflow".to_string(),
            )
        })?;
    if launch.grid_blocks == 0
        || launch.block_threads == 0
        || launch.active_threads != expected_active_threads
        || launch.scratch_words == 0
        || launch.scratch_words != expected_scratch_words
    {
        return Err(GpuPostflightError::InvalidConfiguration(format!(
            "inconsistent field-expression launch: grid={}, block={}, active={}, scratch={} words",
            launch.grid_blocks, launch.block_threads, launch.active_threads, launch.scratch_words,
        )));
    }
    if launch.scratch_words > max_scratch_words {
        return Err(resource_limit(
            "field-expression scratch words",
            launch.scratch_words,
            max_scratch_words,
        ));
    }
    if scratch_bytes > MAX_FIELD_EXPR_SCRATCH_BYTES {
        return Err(resource_limit(
            "field-expression scratch bytes",
            scratch_bytes,
            MAX_FIELD_EXPR_SCRATCH_BYTES,
        ));
    }
    if launch.local_bytes_per_thread > MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD {
        return Err(resource_limit(
            "field-expression local bytes per thread",
            launch.local_bytes_per_thread,
            MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD,
        ));
    }
    if local_bytes > MAX_FIELD_EXPR_LOCAL_BYTES {
        return Err(resource_limit(
            "field-expression total local bytes",
            local_bytes,
            MAX_FIELD_EXPR_LOCAL_BYTES,
        ));
    }
    Ok(launch)
}

fn field_expr_launch_config(
    kernel: ValidatedFieldExprKernelConfig,
    height: usize,
    aux_words_per_thread: usize,
    max_scratch_words: usize,
) -> Result<cuda_abi::FieldExprReplayLaunchConfig, GpuPostflightError> {
    if height == 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "invalid field-expression trace height {height}",
        )));
    }
    if aux_words_per_thread == 0 {
        return Err(GpuPostflightError::InvalidConfiguration(
            "field-expression kernel requires nonzero scratch words per thread".to_string(),
        ));
    }
    let kernel = kernel.0;
    let max_aux_words_per_thread = max_scratch_words / kernel.block_threads;
    if aux_words_per_thread > max_aux_words_per_thread {
        return Err(resource_limit(
            "field-expression scratch words per thread",
            aux_words_per_thread,
            max_aux_words_per_thread,
        ));
    }
    let scratch_words_per_block = kernel.block_threads * aux_words_per_thread;
    let row_blocks = height.div_ceil(kernel.block_threads);
    let scratch_limited_blocks = max_scratch_words / scratch_words_per_block;
    let local_limited_blocks = if kernel.local_bytes_per_thread == 0 {
        usize::MAX
    } else {
        let local_bytes_per_block = kernel.block_threads * kernel.local_bytes_per_thread;
        MAX_FIELD_EXPR_LOCAL_BYTES / local_bytes_per_block
    };
    let grid_blocks = row_blocks
        .min(kernel.max_grid_blocks)
        .min(scratch_limited_blocks)
        .min(local_limited_blocks);
    let active_threads = grid_blocks
        .checked_mul(kernel.block_threads)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch thread count overflow".to_string(),
            )
        })?;
    let scratch_words = active_threads
        .checked_mul(aux_words_per_thread)
        .ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression launch scratch size overflow".to_string(),
            )
        })?;
    validate_launch_config(
        cuda_abi::FieldExprReplayLaunchConfig {
            grid_blocks,
            block_threads: kernel.block_threads,
            scratch_words,
            active_threads,
            local_bytes_per_thread: kernel.local_bytes_per_thread,
        },
        aux_words_per_thread,
        max_scratch_words,
    )
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
    ) -> Result<Self, GpuPostflightError> {
        if range_checker.count.len() != chip.inner.range_checker.count.len() {
            return Err(GpuPostflightError::InvalidConfiguration(
                "field-expression range-count shape mismatch".to_string(),
            ));
        }
        let pointer_max_bits = chip.inner.adapter().pointer_max_bits();
        let timestamp_max_bits = chip.mem_helper.timestamp_max_bits();
        let pointer_max_bits_u32 = u32::try_from(pointer_max_bits).map_err(|_| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression pointer width does not fit u32".to_string(),
            )
        })?;
        let timestamp_max_bits_u32 = u32::try_from(timestamp_max_bits).map_err(|_| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression timestamp width does not fit u32".to_string(),
            )
        })?;
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
                return Err(GpuPostflightError::InvalidConfiguration(
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
            GpuPostflightError::InvalidConfiguration(format!(
                "unsupported device field expression: {error:?}"
            ))
        })?;
        Ok(Self {
            mode: FieldExprReplayMode::Gpu(FieldExprReplayChipGpu::from_serialized(
                chip,
                serialized,
                opcode_base,
                pointer_max_bits_u32,
                timestamp_max_bits_u32,
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
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
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
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    width: usize,
    aux_words_per_thread: usize,
    /// Device-dependent occupancy and kernel attributes, queried once at construction.
    kernel_config: ValidatedFieldExprKernelConfig,
}

impl<const NUM_READS: usize, const BLOCKS: usize> FieldExprReplayChipGpu<NUM_READS, BLOCKS> {
    fn from_serialized(
        chip: &VmChipWrapper<
            F,
            FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>,
        >,
        serialized: openvm_mod_circuit_builder::device_program::SerializedFieldExpr,
        opcode_base: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Self, GpuPostflightError> {
        let num_input_bytes = chip
            .inner
            .num_inputs()
            .checked_mul(chip.inner.expr.program().canonical_num_limbs())
            .ok_or_else(|| {
                GpuPostflightError::InvalidConfiguration(
                    "field-expression input width overflow".to_string(),
                )
            })?;
        let expected_input_bytes = NUM_READS
            .checked_mul(BLOCKS)
            .and_then(|blocks| blocks.checked_mul(openvm_circuit::arch::MEMORY_BLOCK_BYTES))
            .ok_or_else(|| {
                GpuPostflightError::InvalidConfiguration("VecHeap input width overflow".to_string())
            })?;
        if num_input_bytes != expected_input_bytes {
            return Err(GpuPostflightError::InvalidConfiguration(format!(
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
                GpuPostflightError::InvalidConfiguration(
                    "field-expression output width overflow".to_string(),
                )
            })?;
        let expected_output_bytes = BLOCKS
            .checked_mul(openvm_circuit::arch::MEMORY_BLOCK_BYTES)
            .ok_or_else(|| {
                GpuPostflightError::InvalidConfiguration(
                    "VecHeap output width overflow".to_string(),
                )
            })?;
        if num_output_bytes != expected_output_bytes {
            return Err(GpuPostflightError::InvalidConfiguration(format!(
                "field-expression output width {num_output_bytes} does not match VecHeap width {expected_output_bytes}"
            )));
        }
        let adapter_width = Rv64VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS>::width();
        let core_width = BaseAir::<F>::width(&chip.inner.expr);
        if serialized.core_width != core_width {
            return Err(GpuPostflightError::InvalidConfiguration(format!(
                "serialized field-expression width {} does not match AIR width {core_width}",
                serialized.core_width
            )));
        }
        let width = adapter_width.checked_add(core_width).ok_or_else(|| {
            GpuPostflightError::InvalidConfiguration(
                "field-expression trace width overflow".to_string(),
            )
        })?;
        set_device_by_id(range_checker.device_ctx.device_id as i32)?;
        let program = serialized
            .blob
            .as_slice()
            .to_device_on(&range_checker.device_ctx)?;
        let kernel_config = validate_kernel_config(cuda_abi::field_expr_replay_kernel_config::<
            NUM_READS,
            BLOCKS,
        >()?)?;
        Ok(Self {
            range_checker,
            program,
            local_opcodes: chip.inner.local_opcode_idx.clone(),
            opcode_base,
            pointer_max_bits,
            timestamp_max_bits,
            width,
            aux_words_per_thread: serialized.aux_words_per_thread,
            kernel_config,
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
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        let projection = gather_vec_heap_trace_inputs_device::<NUM_READS, BLOCKS>(
            program,
            transcript,
            replay_plan,
            self.opcode_base,
            &self.local_opcodes,
            self.pointer_max_bits as usize,
            device_ctx,
        )?;
        if projection.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        self.generate_from_projection(transcript, projection)
    }

    fn generate_from_projection(
        &self,
        transcript: &GpuPostflightTranscript,
        projection: DeviceVecHeapProjection<NUM_READS, BLOCKS>,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        let (height, _) = checked_trace_shape(
            projection.len(),
            self.width,
            self.timestamp_max_bits as usize,
        )?;
        let trace = DeviceMatrix::<F>::with_capacity_on(height, self.width, device_ctx);
        let delta = DeviceBuffer::with_capacity_on(self.range_checker.count.len(), device_ctx);
        delta.fill_zero_on(device_ctx)?;
        let max_scratch_words = MAX_FIELD_EXPR_SCRATCH_BYTES / size_of::<u32>();
        let launch = field_expr_launch_config(
            self.kernel_config,
            height,
            self.aux_words_per_thread,
            max_scratch_words,
        )?;
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
                self.pointer_max_bits,
                self.timestamp_max_bits,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
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
    use super::*;

    const KERNEL: cuda_abi::FieldExprReplayKernelConfig = cuda_abi::FieldExprReplayKernelConfig {
        max_grid_blocks: 8,
        block_threads: 128,
        local_bytes_per_thread: 32,
    };

    fn kernel() -> ValidatedFieldExprKernelConfig {
        validate_kernel_config(KERNEL).unwrap()
    }

    #[test]
    fn launch_config_caps_grid_by_rows() {
        let launch = field_expr_launch_config(kernel(), 129, 4, 4096).unwrap();
        assert_eq!(launch.grid_blocks, 2);
        assert_eq!(launch.active_threads, 256);
        assert_eq!(launch.scratch_words, 1024);
    }

    #[test]
    fn launch_config_caps_grid_by_scratch() {
        let launch = field_expr_launch_config(kernel(), 4096, 4, 1536).unwrap();
        assert_eq!(launch.grid_blocks, 3);
        assert_eq!(launch.active_threads, 384);
        assert_eq!(launch.scratch_words, 1536);
    }

    #[test]
    fn launch_config_caps_grid_by_occupancy() {
        let launch = field_expr_launch_config(kernel(), 4096, 4, 4096).unwrap();
        assert_eq!(launch.grid_blocks, 8);
        assert_eq!(launch.active_threads, 1024);
        assert_eq!(launch.scratch_words, 4096);
    }

    #[test]
    fn launch_config_rejects_insufficient_scratch() {
        assert!(matches!(
            field_expr_launch_config(kernel(), 1, 4, 0),
            Err(GpuPostflightError::ResourceLimitExceeded {
                resource: "field-expression scratch words per thread",
                requested: 4,
                limit: 0,
            })
        ));
    }

    #[test]
    fn launch_config_caps_grid_by_total_local_memory() {
        let kernel = validate_kernel_config(cuda_abi::FieldExprReplayKernelConfig {
            max_grid_blocks: 1024,
            block_threads: 128,
            local_bytes_per_thread: MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD,
        })
        .unwrap();
        let launch = field_expr_launch_config(
            kernel,
            1 << 20,
            1,
            MAX_FIELD_EXPR_SCRATCH_BYTES / size_of::<u32>(),
        )
        .unwrap();
        assert_eq!(launch.grid_blocks, 512);
        assert_eq!(
            launch.active_threads * launch.local_bytes_per_thread,
            MAX_FIELD_EXPR_LOCAL_BYTES
        );
    }

    #[test]
    fn kernel_config_rejects_invalid_block_size() {
        assert!(matches!(
            validate_kernel_config(cuda_abi::FieldExprReplayKernelConfig {
                max_grid_blocks: 8,
                block_threads: 0,
                local_bytes_per_thread: 32,
            }),
            Err(GpuPostflightError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn kernel_config_rejects_excessive_local_memory() {
        assert!(matches!(
            validate_kernel_config(cuda_abi::FieldExprReplayKernelConfig {
                max_grid_blocks: 8,
                block_threads: 128,
                local_bytes_per_thread: MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD + 1,
            }),
            Err(GpuPostflightError::ResourceLimitExceeded {
                resource: "field-expression local bytes per thread",
                requested,
                limit: MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD,
            }) if requested == MAX_FIELD_EXPR_LOCAL_BYTES_PER_THREAD + 1
        ));
    }
}
