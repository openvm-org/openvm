#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod boundary {
    use super::*;

    extern "C" {
        fn _persistent_boundary_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_raw_records: *const u32,
            num_records: usize,
            d_poseidon2_raw_buffer: *mut std::ffi::c_void,
            d_poseidon2_buffer_idx: *mut u32,
            poseidon2_capacity: usize,
            sbox_regs: usize,
        ) -> i32;

        fn _volatile_boundary_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_raw_records: *const u32,
            num_records: usize,
            d_rc_buffer: *mut std::ffi::c_void,
            rc_num_bins: usize,
            as_max_bits: usize,
            ptr_max_bits: usize,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn persistent_boundary_tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u32>,
        num_records: usize,
        d_poseidon2_raw_buffer: &DeviceBuffer<T>,
        d_poseidon2_buffer_idx: &DeviceBuffer<u32>,
        sbox_regs: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_persistent_boundary_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            d_poseidon2_raw_buffer.as_mut_raw_ptr(),
            d_poseidon2_buffer_idx.as_mut_ptr(),
            d_poseidon2_raw_buffer.len(),
            sbox_regs,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn volatile_boundary_tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u32>,
        num_records: usize,
        d_rc_buffer: &DeviceBuffer<T>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_volatile_boundary_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            d_rc_buffer.as_mut_raw_ptr(),
            d_rc_buffer.len() / 2,
            as_max_bits,
            ptr_max_bits,
        ))
    }
}

pub mod phantom {
    use super::*;

    extern "C" {
        fn _phantom_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            num_records: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_phantom_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
        ))
    }
}

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
