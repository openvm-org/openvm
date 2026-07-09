#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::GpuDeviceCtx};

macro_rules! declare_launcher {
    ($name:ident) => {
        extern "C" {
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
                d_bitwise_lookup: *mut u32,
                bitwise_num_bits: usize,
                d_aux: *mut u32,
                aux_words: usize,
                pointer_max_bits: u32,
                timestamp_max_bits: u32,
                should_finalize: i32,
                d_err: *mut u32,
                stream: openvm_cuda_common::stream::cudaStream_t,
            ) -> i32;
        }
    };
}

declare_launcher!(_field_expr_tracegen_r2_b8);
declare_launcher!(_field_expr_tracegen_r2_b12);
declare_launcher!(_field_expr_tracegen_r2_b16);
declare_launcher!(_field_expr_tracegen_r2_b24);
declare_launcher!(_field_expr_tracegen_r1_b16);
declare_launcher!(_field_expr_tracegen_r1_b24);

pub unsafe fn field_expr_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    rows_used: usize,
    d_blob: &DeviceBuffer<u32>,
    d_records: &DeviceBuffer<u8>,
    rec_stride: usize,
    rec_core_offset: usize,
    d_range_checker: &DeviceBuffer<F>,
    d_bitwise_lookup: &DeviceBuffer<F>,
    bitwise_num_bits: usize,
    d_aux: &DeviceBuffer<u32>,
    aux_words: usize,
    num_reads: usize,
    blocks: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    should_finalize: bool,
    d_err: &DeviceBuffer<u32>,
    device_ctx: &GpuDeviceCtx,
) -> Result<(), CudaError> {
    let launcher = match (num_reads, blocks) {
        (2, 8) => _field_expr_tracegen_r2_b8,
        (2, 12) => _field_expr_tracegen_r2_b12,
        (2, 16) => _field_expr_tracegen_r2_b16,
        (2, 24) => _field_expr_tracegen_r2_b24,
        (1, 16) => _field_expr_tracegen_r1_b16,
        (1, 24) => _field_expr_tracegen_r1_b24,
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
        d_bitwise_lookup.as_mut_ptr() as *mut u32,
        bitwise_num_bits,
        d_aux.as_mut_ptr(),
        aux_words,
        pointer_max_bits,
        timestamp_max_bits,
        should_finalize as i32,
        d_err.as_mut_ptr(),
        device_ctx.stream.as_raw(),
    ))
}
