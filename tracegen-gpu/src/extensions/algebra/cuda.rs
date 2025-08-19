#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_instructions::riscv::RV32_CELL_BITS;
use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod is_eq_cuda {
    use super::*;
    extern "C" {
        fn _modular_is_equal_tracegen(
            d_trace: *mut std::ffi::c_void,
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
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_modulus: &DeviceBuffer<u8>,
        total_limbs: usize,
        num_lanes: usize,
        lane_size: usize,
        d_range_ctr: &DeviceBuffer<T>,
        d_bitwise_lut: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        let record_len = d_records.len();
        let err = _modular_is_equal_tracegen(
            d_trace.as_mut_raw_ptr(),
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
        );
        CudaError::from_result(err)
    }
}
