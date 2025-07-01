#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod castf_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _castf_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            width: u32,
            d_records: *const u8,
            rows_used: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_castf_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
        ))
    }
}

pub mod native_branch_eq_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _native_branch_eq_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            width: u32,
            d_records: *const u8,
            rows_used: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_branch_eq_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
        ))
    }
}
