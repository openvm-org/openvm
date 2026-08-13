#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};
use openvm_riscv_adapters::VecHeapTraceInput;

macro_rules! declare_replay_launcher {
    ($name:ident) => {
        unsafe extern "C" {
            fn $name(
                d_trace: *mut F,
                height: usize,
                width: usize,
                d_instructions: DeviceBufferView,
                pc_base: u32,
                d_program: DeviceBufferView,
                d_memory: DeviceBufferView,
                d_seeds: DeviceBufferView,
                d_predecessors: DeviceBufferView,
                d_steps: DeviceBufferView,
                step_start: usize,
                num_steps: usize,
                d_error: *mut u32,
                opcode_base: u32,
                register_as: u32,
                memory_as: u32,
                d_modulus_limbs: *const u16,
                d_range_checker: *mut u32,
                range_checker_bins: usize,
                pointer_max_bits: u32,
                timestamp_max_bits: u32,
                stream: cudaStream_t,
            ) -> i32;
        }
    };
}

declare_replay_launcher!(_modular_is_eq_replay_tracegen_l4);
declare_replay_launcher!(_modular_is_eq_replay_tracegen_l6);

unsafe extern "C" {
    fn _field_expr_replay_kernel_config(
        num_reads: usize,
        blocks: usize,
        max_grid_blocks: *mut usize,
        block_threads: *mut usize,
        local_bytes_per_thread: *mut usize,
    ) -> i32;

    fn _field_expr_replay_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        num_reads: usize,
        blocks: usize,
        d_projection: *const std::ffi::c_void,
        projection_len: usize,
        d_blob: *const u32,
        blob_words: usize,
        d_range_delta: *mut u32,
        range_bins: usize,
        d_scratch: *mut u32,
        scratch_words: usize,
        aux_words_per_thread: usize,
        grid_blocks: usize,
        block_threads: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _modular_addsub_replay_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        blocks: usize,
        d_projection: *const std::ffi::c_void,
        projection_len: usize,
        d_modulus: *const u8,
        add_local_opcode: u32,
        sub_local_opcode: u32,
        setup_local_opcode: u32,
        d_range_checker: *mut u32,
        range_checker_bins: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _algebra_merge_range_counts(
        destination: *mut u32,
        source: *const u32,
        len: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _vec_heap_replay_gather(
        output: *mut std::ffi::c_void,
        output_len: usize,
        output_start: usize,
        num_reads: usize,
        blocks: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        local_opcode: u32,
        register_as: u32,
        memory_as: u32,
        pointer_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _ec_mul_replay_gather(
        output: *mut std::ffi::c_void,
        output_len: usize,
        output_start: usize,
        blocks: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        is_setup: u32,
        register_as: u32,
        memory_as: u32,
        pointer_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _ec_mul_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        num_limbs: usize,
        blocks: usize,
        d_projection: *const std::ffi::c_void,
        num_instructions: usize,
        d_blob: *const u32,
        blob_words: usize,
        d_vars: *const u32,
        vars_words: usize,
        vars_transposed: bool,
        d_dummy_expr: *const F,
        d_range_counts: *mut u32,
        range_bins: usize,
        d_scratch: *mut u32,
        scratch_words: usize,
        aux_words: usize,
        fill_grid_blocks: usize,
        fill_block_threads: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _ec_mul_projective_generate_vars(
        num_limbs: usize,
        blocks: usize,
        d_projection: *const std::ffi::c_void,
        num_instructions: usize,
        d_blob: *const u32,
        zero_a: bool,
        d_vars: *mut u32,
        vars_words: usize,
        d_projective: *mut u32,
        projective_words: usize,
        d_scratch: *mut u32,
        scratch_words: usize,
        aux_words: usize,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldExprReplayKernelConfig {
    pub max_grid_blocks: usize,
    pub block_threads: usize,
    pub local_bytes_per_thread: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldExprReplayLaunchConfig {
    pub grid_blocks: usize,
    pub block_threads: usize,
    pub scratch_words: usize,
    pub active_threads: usize,
    pub local_bytes_per_thread: usize,
}

/// Queries the device-dependent occupancy cap and kernel attributes once for this chip variant.
pub fn field_expr_replay_kernel_config<const NUM_READS: usize, const BLOCKS: usize>(
) -> Result<FieldExprReplayKernelConfig, CudaError> {
    let mut config = FieldExprReplayKernelConfig {
        max_grid_blocks: 0,
        block_threads: 0,
        local_bytes_per_thread: 0,
    };
    unsafe {
        CudaError::from_result(_field_expr_replay_kernel_config(
            NUM_READS,
            BLOCKS,
            &mut config.max_grid_blocks,
            &mut config.block_threads,
            &mut config.local_bytes_per_thread,
        ))?;
    }
    Ok(config)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn field_expr_replay_tracegen<const NUM_READS: usize, const BLOCKS: usize>(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_projection: &DeviceBuffer<VecHeapTraceInput<NUM_READS, BLOCKS>>,
    d_blob: &DeviceBuffer<u32>,
    d_range_delta: &DeviceBuffer<F>,
    d_scratch: &DeviceBuffer<u32>,
    aux_words_per_thread: usize,
    launch: FieldExprReplayLaunchConfig,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert_eq!(d_scratch.len(), launch.scratch_words);
    CudaError::from_result(_field_expr_replay_tracegen(
        d_trace.as_mut_ptr(),
        height,
        d_trace.len() / height,
        NUM_READS,
        BLOCKS,
        d_projection.as_ptr().cast(),
        d_projection.len(),
        d_blob.as_ptr(),
        d_blob.len(),
        d_range_delta.as_mut_ptr().cast(),
        d_range_delta.len(),
        d_scratch.as_mut_ptr(),
        d_scratch.len(),
        aux_words_per_thread,
        launch.grid_blocks,
        launch.block_threads,
        pointer_max_bits,
        timestamp_max_bits,
        d_error,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn modular_addsub_replay_tracegen<const NUM_READS: usize, const BLOCKS: usize>(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_projection: &DeviceBuffer<VecHeapTraceInput<NUM_READS, BLOCKS>>,
    d_modulus: &DeviceBuffer<u8>,
    add_local_opcode: u32,
    sub_local_opcode: u32,
    setup_local_opcode: u32,
    d_range_checker: &DeviceBuffer<F>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert_eq!(NUM_READS, 2);
    CudaError::from_result(_modular_addsub_replay_tracegen(
        d_trace.as_mut_ptr(),
        height,
        d_trace.len() / height,
        BLOCKS,
        d_projection.as_ptr().cast(),
        d_projection.len(),
        d_modulus.as_ptr(),
        add_local_opcode,
        sub_local_opcode,
        setup_local_opcode,
        d_range_checker.as_mut_ptr().cast(),
        d_range_checker.len(),
        pointer_max_bits,
        timestamp_max_bits,
        d_error,
        stream,
    ))
}

pub unsafe fn merge_range_counts(
    destination: &DeviceBuffer<F>,
    source: &DeviceBuffer<F>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert_eq!(destination.len(), source.len());
    CudaError::from_result(_algebra_merge_range_counts(
        destination.as_mut_ptr().cast(),
        source.as_ptr().cast(),
        source.len(),
        stream,
    ))
}

pub unsafe fn gather_vec_heap<const NUM_READS: usize, const BLOCKS: usize>(
    output: &DeviceBuffer<VecHeapTraceInput<NUM_READS, BLOCKS>>,
    output_start: usize,
    d_instructions: DeviceBufferView,
    pc_base: u32,
    d_program: DeviceBufferView,
    d_memory: DeviceBufferView,
    d_seeds: DeviceBufferView,
    d_predecessors: DeviceBufferView,
    d_steps: DeviceBufferView,
    step_start: usize,
    num_steps: usize,
    expected_opcode: u32,
    local_opcode: u32,
    register_as: u32,
    memory_as: u32,
    pointer_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_vec_heap_replay_gather(
        output.as_mut_ptr().cast(),
        output.len(),
        output_start,
        NUM_READS,
        BLOCKS,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        expected_opcode,
        local_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_error,
        stream,
    ))
}

/// Gathers `EC_MUL` projections from the replayed history.
///
/// `T` is the caller's projection type; this crate cannot name it, since the `EC_MUL` chip lives
/// downstream. The kernel only needs the buffer's address and capacity, and both sides assert the
/// element size independently, so the layout contract is checked without the dependency.
///
/// # Safety
///
/// `T` must have the layout of the kernel's `EcMulTraceInput<BLOCKS>` for the given `blocks`, and
/// the device views must belong to `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_ec_mul<T>(
    output: &DeviceBuffer<T>,
    output_start: usize,
    blocks: usize,
    d_instructions: DeviceBufferView,
    pc_base: u32,
    d_program: DeviceBufferView,
    d_memory: DeviceBufferView,
    d_seeds: DeviceBufferView,
    d_predecessors: DeviceBufferView,
    d_steps: DeviceBufferView,
    step_start: usize,
    num_steps: usize,
    expected_opcode: u32,
    is_setup: bool,
    register_as: u32,
    memory_as: u32,
    pointer_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_ec_mul_replay_gather(
        output.as_mut_ptr().cast(),
        output.len(),
        output_start,
        blocks,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        expected_opcode,
        u32::from(is_setup),
        register_as,
        memory_as,
        pointer_max_bits,
        d_error,
        stream,
    ))
}

pub unsafe fn replay_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    d_instructions: DeviceBufferView,
    pc_base: u32,
    d_program: DeviceBufferView,
    d_memory: DeviceBufferView,
    d_seeds: DeviceBufferView,
    d_predecessors: DeviceBufferView,
    d_steps: DeviceBufferView,
    step_start: usize,
    num_steps: usize,
    d_error: *mut u32,
    opcode_base: u32,
    register_as: u32,
    memory_as: u32,
    d_modulus_limbs: &DeviceBuffer<u16>,
    d_range_checker: &DeviceBuffer<F>,
    num_lanes: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let launcher = match num_lanes {
        4 => _modular_is_eq_replay_tracegen_l4,
        6 => _modular_is_eq_replay_tracegen_l6,
        _ => panic!("unsupported ModularIsEqual num_lanes {num_lanes}"),
    };
    CudaError::from_result(launcher(
        d_trace.as_mut_ptr(),
        height,
        d_trace.len() / height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        d_error,
        opcode_base,
        register_as,
        memory_as,
        d_modulus_limbs.as_ptr(),
        d_range_checker.as_mut_ptr() as *mut u32,
        d_range_checker.len(),
        pointer_max_bits,
        timestamp_max_bits,
        stream,
    ))
}

/// Launch dimensions for the row-filling pass, whose scratch is bounded by its thread count.
#[derive(Clone, Copy, Debug)]
pub struct EcMulFillLaunchConfig {
    pub grid_blocks: usize,
    pub block_threads: usize,
    pub scratch_words: usize,
}

/// Generates the exact field-expression saved variables for the secp256k1 EC MUL specialization.
///
/// # Safety
///
/// `T` must match the selected device `EcMulTraceInput<BLOCKS>` layout and all buffers must share
/// `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ec_mul_projective_generate_vars<T>(
    num_limbs: usize,
    blocks: usize,
    d_projection: &DeviceBuffer<T>,
    d_blob: &DeviceBuffer<u32>,
    zero_a: bool,
    d_vars: &DeviceBuffer<u32>,
    d_projective: &DeviceBuffer<u32>,
    d_scratch: &DeviceBuffer<u32>,
    aux_words: usize,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_ec_mul_projective_generate_vars(
        num_limbs,
        blocks,
        d_projection.as_ptr().cast(),
        d_projection.len(),
        d_blob.as_ptr(),
        zero_a,
        d_vars.as_mut_ptr(),
        d_vars.len(),
        d_projective.as_mut_ptr(),
        d_projective.len(),
        d_scratch.as_mut_ptr(),
        d_scratch.len(),
        aux_words,
        d_error,
        stream,
    ))
}

/// Generates one curve's `EC_MUL` trace from already-gathered projections.
///
/// `T` is the caller's projection type; this crate cannot name it, since the `EC_MUL` chip lives
/// downstream. Both sides assert the element size independently, as for [`gather_ec_mul`].
///
/// # Safety
///
/// `T` must have the layout of the kernel's `EcMulTraceInput<BLOCKS>` for the given `blocks`, and
/// every device view must belong to `stream`'s context.
pub unsafe fn ec_mul_tracegen<T>(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    num_limbs: usize,
    blocks: usize,
    d_projection: &DeviceBuffer<T>,
    d_blob: &DeviceBuffer<u32>,
    d_vars: &DeviceBuffer<u32>,
    vars_transposed: bool,
    d_dummy_expr: &DeviceBuffer<F>,
    d_range_counts: &DeviceBuffer<F>,
    d_scratch: &DeviceBuffer<u32>,
    aux_words: usize,
    launch: EcMulFillLaunchConfig,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_ec_mul_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        num_limbs,
        blocks,
        d_projection.as_ptr().cast(),
        d_projection.len(),
        d_blob.as_ptr(),
        d_blob.len(),
        d_vars.as_ptr(),
        d_vars.len(),
        vars_transposed,
        d_dummy_expr.as_ptr(),
        d_range_counts.as_mut_ptr().cast(),
        d_range_counts.len(),
        d_scratch.as_mut_ptr(),
        launch.scratch_words,
        aux_words,
        launch.grid_blocks,
        launch.block_threads,
        pointer_max_bits,
        timestamp_max_bits,
        d_error,
        stream,
    ))
}
