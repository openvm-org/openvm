#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};

macro_rules! declare_is_eq_launcher {
    ($name:ident) => {
        extern "C" {
            fn $name(
                d_trace: *mut F,
                height: usize,
                rows_used: usize,
                d_records: *const u8,
                rec_stride: usize,
                rec_core_offset: usize,
                d_modulus_limbs: *const u16,
                d_range_checker: *mut u32,
                rc_bins: usize,
                pointer_max_bits: u32,
                timestamp_max_bits: u32,
                stream: cudaStream_t,
            ) -> i32;
        }
    };
}
declare_is_eq_launcher!(_modular_is_eq_tracegen_l4);
declare_is_eq_launcher!(_modular_is_eq_tracegen_l6);

pub unsafe fn tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    rows_used: usize,
    d_records: &DeviceBuffer<u8>,
    rec_stride: usize,
    rec_core_offset: usize,
    d_modulus_limbs: &DeviceBuffer<u16>,
    d_range_checker: &DeviceBuffer<F>,
    num_lanes: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let launcher = match num_lanes {
        4 => _modular_is_eq_tracegen_l4,
        6 => _modular_is_eq_tracegen_l6,
        _ => panic!("unsupported ModularIsEqual num_lanes {num_lanes}"),
    };
    CudaError::from_result(launcher(
        d_trace.as_mut_ptr(),
        height,
        rows_used,
        d_records.as_ptr(),
        rec_stride,
        rec_core_offset,
        d_modulus_limbs.as_ptr(),
        d_range_checker.as_mut_ptr() as *mut u32,
        d_range_checker.len(),
        pointer_max_bits,
        timestamp_max_bits,
        stream,
    ))
}
