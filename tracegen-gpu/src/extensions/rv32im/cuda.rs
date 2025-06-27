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
            range_checker_max_bins: usize,
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

pub mod hintstore {
    use super::super::hintstore::OffsetInfo;
    use super::*;

    unsafe extern "C" {
        unsafe fn _hintstore_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: u32,
            d_record_offsets: *const OffsetInfo,
            pointer_max_bits: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_hintstore_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            rows_used,
            d_record_offsets.as_ptr(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() / 2,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
        ))
    }
}

pub mod jalr {
    use super::*;

    extern "C" {
        fn _jalr_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_max_bins: usize,
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
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_jalr_tracegen(
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

pub mod less_than {
    use super::*;

    extern "C" {
        fn _rv32_less_than_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: usize,
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
        CudaError::from_result(_rv32_less_than_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
        ))
    }
}

pub mod mul {
    use super::*;

    extern "C" {
        fn _mul_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<T>,
        range_bins: usize,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_mul_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            d_records.len(),
            d_range.as_ptr() as *const u32,
            range_bins,
        ))
    }
}

pub mod shift {
    use super::*;

    extern "C" {
        fn _rv32_shift_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: usize,
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
        CudaError::from_result(_rv32_shift_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
        ))
    }
}

pub mod alu {
    use super::*;
    extern "C" {
        fn _alu_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<T>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: usize,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_alu_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker.as_ptr() as *const u32,
            range_bins,
            d_bitwise_lookup.as_ptr() as *const u32,
            bitwise_num_bits,
        ))
    }
}

pub mod loadstore_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _rv32_load_store_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            pointer_max_bits: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv32_load_store_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32 / 2,
        ))
    }
}

pub mod load_sign_extend_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _rv32_load_sign_extend_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            pointer_max_bits: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv32_load_sign_extend_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32 / 2,
        ))
    }
}

pub mod jal_lui {
    use super::*;

    extern "C" {
        fn _jal_lui_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_max_bins: usize,
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
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_jal_lui_tracegen(
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
