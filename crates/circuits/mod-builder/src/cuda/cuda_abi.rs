#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::GpuDeviceCtx};

macro_rules! declare_launcher {
    ($name:ident, $max_grid_blocks:ident) => {
        extern "C" {
            fn $max_grid_blocks(out_max_grid_blocks: *mut u32) -> i32;
            fn $name(
                d_trace: *mut F,
                height: usize,
                rows_used: usize,
                d_blob: *const u32,
                d_records: *const u8,
                rec_stride: usize,
                rec_core_offset: usize,
                d_range_checker: *mut u32,
                rc_bins: usize,
                d_aux: *mut u32,
                aux_words: usize,
                pointer_max_bits: u32,
                timestamp_max_bits: u32,
                should_finalize: i32,
                d_err: *mut u32,
                grid_blocks: u32,
                stream: openvm_cuda_common::stream::cudaStream_t,
            ) -> i32;
        }
    };
}

declare_launcher!(
    _field_expr_tracegen_r2_b4,
    _field_expr_max_grid_blocks_r2_b4
);
declare_launcher!(
    _field_expr_tracegen_r2_b6,
    _field_expr_max_grid_blocks_r2_b6
);
declare_launcher!(
    _field_expr_tracegen_r2_b8,
    _field_expr_max_grid_blocks_r2_b8
);
declare_launcher!(
    _field_expr_tracegen_r2_b12,
    _field_expr_max_grid_blocks_r2_b12
);
declare_launcher!(
    _field_expr_tracegen_r1_b8,
    _field_expr_max_grid_blocks_r1_b8
);
declare_launcher!(
    _field_expr_tracegen_r1_b12,
    _field_expr_max_grid_blocks_r1_b12
);

/// One-time query of the co-resident grid cap (SM count x per-SM occupancy) for the
/// (num_reads, blocks) kernel variant. Must be called with the chip's device current;
/// the chip caches the result and sizes both `d_aux` and the launch grid from it.
pub fn max_grid_blocks(num_reads: usize, blocks: usize) -> Result<u32, CudaError> {
    let query = match (num_reads, blocks) {
        (2, 4) => _field_expr_max_grid_blocks_r2_b4,
        (2, 6) => _field_expr_max_grid_blocks_r2_b6,
        (2, 8) => _field_expr_max_grid_blocks_r2_b8,
        (2, 12) => _field_expr_max_grid_blocks_r2_b12,
        (1, 8) => _field_expr_max_grid_blocks_r1_b8,
        (1, 12) => _field_expr_max_grid_blocks_r1_b12,
        _ => panic!("unsupported (num_reads, blocks) = ({num_reads}, {blocks})"),
    };
    let mut max_grid_blocks = 0u32;
    unsafe { CudaError::from_result(query(&mut max_grid_blocks))? };
    Ok(max_grid_blocks)
}

pub unsafe fn field_expr_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    rows_used: usize,
    d_blob: &DeviceBuffer<u32>,
    d_records: &DeviceBuffer<u8>,
    rec_stride: usize,
    rec_core_offset: usize,
    d_range_checker: &DeviceBuffer<F>,
    d_aux: &DeviceBuffer<u32>,
    aux_words: usize,
    num_reads: usize,
    blocks: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    should_finalize: bool,
    d_err: &DeviceBuffer<u32>,
    grid_blocks: u32,
    device_ctx: &GpuDeviceCtx,
) -> Result<(), CudaError> {
    let launcher = match (num_reads, blocks) {
        (2, 4) => _field_expr_tracegen_r2_b4,
        (2, 6) => _field_expr_tracegen_r2_b6,
        (2, 8) => _field_expr_tracegen_r2_b8,
        (2, 12) => _field_expr_tracegen_r2_b12,
        (1, 8) => _field_expr_tracegen_r1_b8,
        (1, 12) => _field_expr_tracegen_r1_b12,
        _ => panic!("unsupported (num_reads, blocks) = ({num_reads}, {blocks})"),
    };
    CudaError::from_result(launcher(
        d_trace.as_mut_ptr(),
        height,
        rows_used,
        d_blob.as_ptr(),
        d_records.as_ptr(),
        rec_stride,
        rec_core_offset,
        d_range_checker.as_mut_ptr() as *mut u32,
        d_range_checker.len(),
        d_aux.as_mut_ptr(),
        aux_words,
        pointer_max_bits,
        timestamp_max_bits,
        should_finalize as i32,
        d_err.as_mut_ptr(),
        grid_blocks,
        device_ctx.stream.as_raw(),
    ))
}
