#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};

extern "C" {
    fn _field_expr_tracegen(
        d_trace_core: *mut F,
        height: usize,
        rows_used: usize,
        d_blob: *const u32,
        d_records: *const u8,
        rec_stride: usize,
        rec_core_offset: usize,
        d_range_checker: *mut u32,
        d_aux: *mut u32,
        aux_words: usize,
        should_finalize: i32,
        d_err: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
}

pub unsafe fn field_expr_tracegen(
    d_trace: &DeviceBuffer<F>,
    adapter_width: usize,
    height: usize,
    rows_used: usize,
    d_blob: &DeviceBuffer<u32>,
    d_records: &DeviceBuffer<u8>,
    rec_stride: usize,
    rec_core_offset: usize,
    d_range_checker: &DeviceBuffer<F>,
    d_aux: &DeviceBuffer<u32>,
    aux_words: usize,
    should_finalize: bool,
    d_err: &DeviceBuffer<u32>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    // Core columns are column-major starting after the adapter columns:
    // pointer offset = adapter_width * height.
    CudaError::from_result(_field_expr_tracegen(
        d_trace.as_mut_ptr().add(adapter_width * height),
        height,
        rows_used,
        d_blob.as_ptr(),
        d_records.as_ptr(),
        rec_stride,
        rec_core_offset,
        d_range_checker.as_mut_ptr() as *mut u32,
        d_aux.as_mut_ptr(),
        aux_words,
        should_finalize as i32,
        d_err.as_mut_ptr(),
        stream,
    ))
}
