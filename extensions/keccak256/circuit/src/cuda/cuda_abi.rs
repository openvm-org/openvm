#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod keccak256 {
    use super::*;

    extern "C" {
        fn _keccakf_kernel(
            d_records: *const u8,
            num_records: usize,
            d_record_offsets: *const usize,
            d_block_offsets: *const u32,
            total_num_blocks: u32,
            d_states: *mut u64,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
        ) -> i32;

        fn _keccak256_p3_tracegen(
            d_trace: *mut F,
            height: usize,
            total_num_blocks: u32,
            d_states: *mut u64,
        ) -> i32;

        fn _keccak256_tracegen(
            d_trace: *mut F,
            height: usize,
            d_records: *const u8,
            num_records: usize,
            d_record_offsets: *const usize,
            d_block_offsets: *const u32,
            d_block_to_record_idx: *const u32,
            total_num_blocks: u32,
            d_states: *const u64,
            rows_used: usize,
            ptr_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn keccakf(
        d_records: &DeviceBuffer<u8>,
        records_num: usize,
        d_record_offsets: &DeviceBuffer<usize>,
        d_block_offsets: &DeviceBuffer<u32>,
        total_num_blocks: u32,
        d_states: &DeviceBuffer<u64>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_keccakf_kernel(
            d_records.as_ptr(),
            records_num,
            d_record_offsets.as_ptr(),
            d_block_offsets.as_ptr(),
            total_num_blocks,
            d_states.as_mut_ptr(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
        ))
    }

    pub unsafe fn p3_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        total_num_blocks: u32,
        d_states: &DeviceBuffer<u64>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_keccak256_p3_tracegen(
            d_trace.as_mut_ptr(),
            height,
            total_num_blocks,
            d_states.as_mut_ptr(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        records_num: usize,
        d_record_offsets: &DeviceBuffer<usize>,
        d_block_offsets: &DeviceBuffer<u32>,
        d_block_to_record_idx: &DeviceBuffer<u32>,
        total_num_blocks: u32,
        d_states: &DeviceBuffer<u64>,
        rows_used: usize,
        ptr_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_keccak256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_records.as_ptr(),
            records_num,
            d_record_offsets.as_ptr(),
            d_block_offsets.as_ptr(),
            d_block_to_record_idx.as_ptr(),
            total_num_blocks,
            d_states.as_ptr(),
            rows_used,
            ptr_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}
