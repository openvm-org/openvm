use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
};

pub mod xorin {
    use super::*;

    extern "C" {
        fn _xorin_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    /// # Safety
    /// All device buffers must be valid and properly allocated.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_xorin_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        ))
    }
}

/// FFI bindings for the new KeccakfOpChip GPU kernel
pub mod keccakf_op {
    use super::*;

    extern "C" {
        fn _keccakf_op_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *const u32,
            bitwise_num_bits: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    /// # Safety
    /// All device buffers must be valid and properly allocated.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_op_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        ))
    }
}

/// FFI bindings for the new KeccakfPermChip GPU kernel
pub mod keccakf_perm {
    use super::*;

    extern "C" {
        fn _keccakf_perm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            num_records: usize,
        ) -> i32;
    }

    /// # Safety
    /// All device buffers must be valid and properly allocated.
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_perm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            num_records,
        ))
    }
}
