#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::primitives::exp_bits_len::ExpBitsLenRecord;

extern "C" {
    fn _pow_checker_tracegen(
        d_pow_count: *const u32,
        d_range_count: *const u32,
        d_trace: *mut F,
        n: usize,
    ) -> i32;

    fn _range_checker_recursion_tracegen(
        d_count: *const u32,
        d_trace: *mut F,
        num_bits: usize,
    ) -> i32;

    fn _exp_bits_len_tracegen(
        d_requests: *const ExpBitsLenRecord,
        num_requests: usize,
        d_trace: *mut F,
        height: usize,
        num_valid_rows: usize,
    ) -> i32;
}

pub unsafe fn pow_checker_tracegen(
    d_pow_count: *const u32,
    d_range_count: *const u32,
    d_trace: &DeviceBuffer<F>,
    n: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_pow_checker_tracegen(
        d_pow_count,
        d_range_count,
        d_trace.as_mut_ptr(),
        n,
    ))
}

pub unsafe fn range_checker_tracegen(
    d_count: *const u32,
    d_trace: &DeviceBuffer<F>,
    num_bits: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_range_checker_recursion_tracegen(
        d_count,
        d_trace.as_mut_ptr(),
        num_bits,
    ))
}

#[tracing::instrument(name = "exp_bits_len_tracegen", skip_all)]
pub unsafe fn exp_bits_len_tracegen(
    d_requests: *const ExpBitsLenRecord,
    num_requests: usize,
    d_trace: &DeviceBuffer<F>,
    height: usize,
    num_valid_rows: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_exp_bits_len_tracegen(
        d_requests,
        num_requests,
        d_trace.as_mut_ptr(),
        height,
        num_valid_rows,
    ))
}
