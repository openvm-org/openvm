#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::{
    cuda::{
        d_buffer::{DeviceBuffer, DeviceBufferView},
        error::CudaError,
    },
    prelude::F,
};

pub mod execution_testing {
    use super::*;

    unsafe extern "C" {
        unsafe fn _execution_testing_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_execution_testing_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
        ))
    }
}

pub mod memory_testing {
    use super::*;

    unsafe extern "C" {
        unsafe fn _memory_testing_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const F,
            num_records: usize,
            block_size: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<F>,
        num_records: usize,
        block_size: usize,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        assert!(height >= num_records);
        assert!(block_size.is_power_of_two());
        CudaError::from_result(_memory_testing_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            block_size,
        ))
    }
}

pub mod program_testing {
    use super::*;

    unsafe extern "C" {
        unsafe fn _program_testing_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            num_records: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        assert!(height >= num_records);
        CudaError::from_result(_program_testing_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
        ))
    }
}
