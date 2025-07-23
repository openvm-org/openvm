#![allow(clippy::missing_safety_doc)]

use openvm_stark_backend::prover::hal::MatrixDimensions;
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
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_castf_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
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
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_branch_eq_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod field_arithmetic_cuda {
    use std::ffi::c_void;

    use super::*;

    extern "C" {
        fn _field_arithmetic_tracegen(
            d_trace: *mut c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: *const u32,
        range_bins: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = _field_arithmetic_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            d_records.len(),
            d_range_checker,
            range_bins,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }
}

pub mod field_extension_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _field_extension_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            width: u32,
            d_records: *const u8,
            rows_used: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_field_extension_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod fri_cuda {
    use super::*;
    use crate::extensions::native::RowInfo;

    unsafe extern "C" {
        unsafe fn _fri_reduced_opening_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            d_records: *const u8,
            rows_used: u32,
            d_record_info: *const RowInfo,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_record_info: &DeviceBuffer<RowInfo>,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fri_reduced_opening_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            d_records.as_ptr(),
            rows_used,
            d_record_info.as_ptr(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod poseidon2_cuda {
    use stark_backend_gpu::base::DeviceMatrix;

    use super::*;

    unsafe extern "C" {
        fn _inplace_native_poseidon2_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            num_records: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            sbox_regs: usize,
            timestamp_max_bits: u32,
        ) -> i32;

        fn _native_poseidon2_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const std::ffi::c_void,
            num_records: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            sbox_regs: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn inplace_tracegen<T>(
        d_trace: &DeviceMatrix<T>,
        num_records: usize,
        d_range_checker: &DeviceBuffer<T>,
        sbox_regs: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_inplace_native_poseidon2_tracegen(
            d_trace.buffer().as_mut_raw_ptr(),
            d_trace.height(),
            d_trace.width(),
            num_records,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            sbox_regs,
            timestamp_max_bits,
        ))
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceMatrix<T>,
        d_records: &DeviceBuffer<T>,
        num_records: usize,
        d_range_checker: &DeviceBuffer<T>,
        sbox_regs: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_poseidon2_tracegen(
            d_trace.buffer().as_mut_raw_ptr(),
            d_trace.height(),
            d_trace.width(),
            d_records.as_raw_ptr(),
            num_records,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            sbox_regs,
            timestamp_max_bits,
        ))
    }
}

pub mod native_loadstore_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _native_loadstore_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            width: u32,
            d_records: *const u8,
            rows_used: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            num_cells: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
        num_cells: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_loadstore_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            num_cells,
            timestamp_max_bits,
        ))
    }
}

pub mod native_jal_rangecheck_cuda {
    use super::*;

    unsafe extern "C" {
        unsafe fn _native_jal_rangecheck_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: u32,
            width: u32,
            d_records: *const u8,
            rows_used: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: u32,
        width: u32,
        d_records: &DeviceBuffer<u8>,
        rows_used: u32,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_jal_rangecheck_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}
