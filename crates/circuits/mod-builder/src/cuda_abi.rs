#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod field_expression {

    use super::*;
    extern "C" {
        fn _field_expression_tracegen(
            d_records: *const u8,
            d_trace: *mut std::ffi::c_void,
            d_meta: *const std::ffi::c_void,
            num_records: usize,
            record_stride: usize,
            width: usize,
            height: usize,
            d_range_checker: *const u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            d_workspace: *const u8,
            workspace_per_thread: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T, M>(
        d_records: &DeviceBuffer<u8>,
        d_trace: &DeviceBuffer<T>,
        d_meta: &DeviceBuffer<M>,
        num_records: usize,
        record_stride: usize,
        width: usize,
        height: usize,
        d_range_checker: &DeviceBuffer<T>,
        d_bitwise_lookup: &DeviceBuffer<T>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        d_workspace: *const u8,
        workspace_per_thread: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_field_expression_tracegen(
            d_records.as_ptr(),
            d_trace.as_mut_raw_ptr(),
            d_meta.as_ptr() as *const std::ffi::c_void,
            num_records,
            record_stride,
            width,
            height,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
            d_workspace,
            workspace_per_thread,
        ))
    }
}
