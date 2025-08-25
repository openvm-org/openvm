#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
};

pub mod castf_cuda {
    use super::*;

    extern "C" {
        pub fn _castf_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_castf_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod native_branch_eq_cuda {
    use super::*;

    extern "C" {
        pub fn _native_branch_eq_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_branch_eq_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod field_arithmetic_cuda {
    use super::*;

    extern "C" {
        fn _field_arithmetic_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: *const u32,
        range_bins: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = _field_arithmetic_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker,
            range_bins,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }
}

pub mod field_extension_cuda {
    use super::*;

    extern "C" {
        pub fn _field_extension_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_field_extension_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod fri_cuda {
    use super::*;
    use crate::fri::RowInfo;

    extern "C" {
        pub fn _fri_reduced_opening_tracegen(
            d_trace: *mut F,
            height: usize,
            d_records: *const u8,
            rows_used: usize,
            d_record_info: *const RowInfo,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        d_record_info: &DeviceBuffer<RowInfo>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fri_reduced_opening_tracegen(
            d_trace.as_mut_ptr(),
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
    use super::*;

    extern "C" {
        pub fn _native_poseidon2_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            d_chunk_start: *const u32,
            num_chunks: u32,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            sbox_regs: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        d_chunk_start: &DeviceBuffer<u32>,
        num_chunks: u32,
        d_range_checker: &DeviceBuffer<F>,
        sbox_regs: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_poseidon2_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_ptr(),
            rows_used,
            d_chunk_start.as_ptr(),
            num_chunks,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            sbox_regs,
            timestamp_max_bits,
        ))
    }
}

pub mod native_loadstore_cuda {
    use super::*;

    extern "C" {
        pub fn _native_loadstore_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            num_cells: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        num_cells: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_loadstore_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            num_cells,
            timestamp_max_bits,
        ))
    }
}

pub mod native_jal_rangecheck_cuda {
    use super::*;

    extern "C" {
        pub fn _native_jal_rangecheck_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_max_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_native_jal_rangecheck_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}
