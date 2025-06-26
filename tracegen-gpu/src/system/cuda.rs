#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod poseidon2 {
    use super::*;

    extern "C" {
        fn _system_poseidon2_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *mut std::ffi::c_void,
            d_counts: *mut u32,
            num_records: usize,
            sbox_regs: usize,
        ) -> i32;

        fn _system_poseidon2_deduplicate_records(
            d_records: *mut std::ffi::c_void,
            d_counts: *mut u32,
            num_records: *mut usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<T>,
        d_counts: &DeviceBuffer<u32>,
        num_records: usize,
        sbox_regs: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_mut_raw_ptr(),
            d_counts.as_mut_ptr(),
            num_records,
            sbox_regs,
        ))
    }

    pub unsafe fn deduplicate_records<T>(
        d_records: &DeviceBuffer<T>,
        d_counts: &DeviceBuffer<u32>,
        num_records: &mut usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_deduplicate_records(
            d_records.as_mut_raw_ptr(),
            d_counts.as_mut_ptr(),
            num_records as *mut usize,
        ))
    }
}
