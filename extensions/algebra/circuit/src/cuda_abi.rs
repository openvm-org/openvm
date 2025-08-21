#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_backend::types::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_instructions::riscv::RV32_CELL_BITS;

pub mod is_eq_cuda {
    use super::*;

    extern "C" {
        fn _modular_is_equal_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_modulus: *const u8,
            total_limbs: usize,
            num_lanes: usize,
            lane_size: usize,
            d_range_ctr: *const u32,
            range_bins: usize,
            d_bitwise_lut: *const u32,
            bitwise_num_bits: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_modulus: &DeviceBuffer<u8>,
        total_limbs: usize,
        num_lanes: usize,
        lane_size: usize,
        d_range_ctr: &DeviceBuffer<F>,
        d_bitwise_lut: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        let record_len = d_records.len();
        let err = _modular_is_equal_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_ptr(),
            record_len,
            d_modulus.as_ptr(),
            total_limbs,
            num_lanes,
            lane_size,
            d_range_ctr.as_ptr() as *const u32,
            d_range_ctr.len(),
            d_bitwise_lut.as_ptr() as *const u32,
            RV32_CELL_BITS,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(err)
    }
}
