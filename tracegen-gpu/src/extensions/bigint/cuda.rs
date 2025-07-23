#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

use crate::UInt2;

pub mod alu256 {
    use super::*;

    extern "C" {
        fn _alu256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_alu256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod beq256 {
    use super::*;

    extern "C" {
        fn _branch_equal256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_branch_equal256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod lt256 {
    use super::*;

    extern "C" {
        fn _less_than256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_less_than256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod blt256 {
    use super::*;

    extern "C" {
        fn _branch_less_than256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_branch_less_than256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod shift256 {
    use super::*;

    extern "C" {
        fn _shift256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_shift256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod mul256 {
    use super::*;

    extern "C" {
        fn _multiplication256_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            d_range_tuple: *const u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
        d_range_tuple: &DeviceBuffer<T>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_multiplication256_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            d_range_tuple.as_mut_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
        ))
    }
}
