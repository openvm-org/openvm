#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod auipc {
    use super::*;

    extern "C" {
        fn _auipc_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_max_bits: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_auipc_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() / 2,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
        ))
    }
}
