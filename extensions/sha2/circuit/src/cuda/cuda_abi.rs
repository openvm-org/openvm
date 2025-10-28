#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod sha256 {
    use super::*;

    extern "C" {
        fn launch_sha256_hash_computation(
            d_records: *const u8,
            num_records: usize,
            d_record_offsets: *const usize,
            d_block_offsets: *const u32,
            d_prev_hashes: *mut u32,
            total_num_blocks: u32,
        ) -> i32;

        fn launch_sha256_first_pass_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            d_record_offsets: *const usize,
            d_block_offsets: *const u32,
            d_block_to_record_idx: *const u32,
            total_num_blocks: u32,
            d_prev_hashes: *const u32,
            ptr_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn launch_sha256_second_pass_dependencies(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
        ) -> i32;

        fn launch_sha256_fill_invalid_rows(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
        ) -> i32;
    }

    pub unsafe fn sha256_hash_computation(
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        d_record_offsets: &DeviceBuffer<usize>,
        d_block_offsets: &DeviceBuffer<u32>,
        d_prev_hashes: &DeviceBuffer<u32>,
        num_blocks: u32,
    ) -> Result<(), CudaError> {
        let result = launch_sha256_hash_computation(
            d_records.as_ptr(),
            num_records,
            d_record_offsets.as_ptr(),
            d_block_offsets.as_ptr(),
            d_prev_hashes.as_mut_ptr(),
            num_blocks,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sha256_first_pass_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        d_record_offsets: &DeviceBuffer<usize>,
        d_block_offsets: &DeviceBuffer<u32>,
        d_block_to_record_idx: &DeviceBuffer<u32>,
        total_num_blocks: u32,
        d_prev_hashes: &DeviceBuffer<u32>,
        ptr_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_sha256_first_pass_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_records.as_ptr(),
            num_records,
            d_record_offsets.as_ptr(),
            d_block_offsets.as_ptr(),
            d_block_to_record_idx.as_ptr(),
            total_num_blocks,
            d_prev_hashes.as_ptr(),
            ptr_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }

    pub unsafe fn sha256_second_pass_dependencies(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
    ) -> Result<(), CudaError> {
        let result =
            launch_sha256_second_pass_dependencies(d_trace.as_mut_ptr(), height, rows_used);
        CudaError::from_result(result)
    }

    pub unsafe fn sha256_fill_invalid_rows(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
    ) -> Result<(), CudaError> {
        let result = launch_sha256_fill_invalid_rows(d_trace.as_mut_ptr(), height, rows_used);
        CudaError::from_result(result)
    }
}
