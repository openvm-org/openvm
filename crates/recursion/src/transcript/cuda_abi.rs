#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

extern "C" {
    fn _poseidon2_tracegen(
        d_trace: *mut F,
        height: usize,
        width: usize,
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        sbox_regs: usize,
    ) -> i32;

    fn _poseidon2_deduplicate_records_get_temp_bytes(
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        d_num_records: *mut usize,
        h_temp_bytes_out: *mut usize,
    ) -> i32;

    fn _poseidon2_deduplicate_records(
        d_records: *mut F,
        d_counts: *mut u32,
        num_records: usize,
        d_num_records: *mut usize,
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
    ) -> i32;
}

pub unsafe fn poseidon2_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    width: usize,
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    sbox_regs: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_tracegen(
        d_trace.as_mut_ptr(),
        height,
        width,
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        sbox_regs,
    ))
}

pub unsafe fn poseidon2_deduplicate_records_get_temp_bytes(
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    d_num_records: &DeviceBuffer<usize>,
    h_temp_bytes_out: &mut usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_deduplicate_records_get_temp_bytes(
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        d_num_records.as_mut_ptr(),
        h_temp_bytes_out,
    ))
}

pub unsafe fn poseidon2_deduplicate_records(
    d_records: &DeviceBuffer<F>,
    d_counts: &DeviceBuffer<u32>,
    num_records: usize,
    d_num_records: &DeviceBuffer<usize>,
    d_temp_storage: &DeviceBuffer<u8>,
    temp_storage_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_poseidon2_deduplicate_records(
        d_records.as_mut_ptr(),
        d_counts.as_mut_ptr(),
        num_records,
        d_num_records.as_mut_ptr(),
        d_temp_storage.as_mut_ptr() as *mut std::ffi::c_void,
        temp_storage_bytes,
    ))
}
