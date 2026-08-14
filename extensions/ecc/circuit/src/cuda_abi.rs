//! Host-side launchers for the `EC_MUL` kernels in `cuda/`.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

unsafe extern "C" {
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

/// Gathers `EC_MUL` projections from the replayed history.
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

/// Launch dimensions for the row-filling pass, whose scratch is bounded by its thread count.
#[derive(Clone, Copy, Debug)]
pub struct EcMulFillLaunchConfig {
    pub grid_blocks: usize,
    pub block_threads: usize,
    pub scratch_words: usize,
}

/// Generates the exact field-expression saved variables for a supported EC MUL shape.
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
