#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod bitwise_op_lookup {
    use super::*;

    extern "C" {
        fn _bitwise_op_lookup_tracegen(
            d_count: *const u32,
            d_trace: *mut std::ffi::c_void,
            num_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_count: &DeviceBuffer<T>,
        d_trace: &DeviceBuffer<T>,
        num_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_bitwise_op_lookup_tracegen(
            d_count.as_ptr() as *const u32,
            d_trace.as_mut_raw_ptr(),
            num_bits,
        ))
    }
}

pub mod var_range {
    use super::*;

    extern "C" {
        fn _range_checker_tracegen(
            d_count: *const u32,
            d_trace: *mut std::ffi::c_void,
            num_bins: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_count: &DeviceBuffer<T>,
        d_trace: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_checker_tracegen(
            d_count.as_ptr() as *const u32,
            d_trace.as_mut_raw_ptr(),
            d_count.len(),
        ))
    }
}

pub mod range_tuple {
    use super::*;

    extern "C" {
        fn _range_tuple_checker_tracegen(
            d_count: *const u32,
            d_trace: *mut std::ffi::c_void,
            num_bins: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_count: &DeviceBuffer<T>,
        d_trace: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_tuple_checker_tracegen(
            d_count.as_ptr() as *const u32,
            d_trace.as_mut_raw_ptr(),
            d_count.len(),
        ))
    }
}
