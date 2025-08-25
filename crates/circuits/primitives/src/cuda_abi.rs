#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod bitwise_op_lookup {
    #[cfg(test)]
    use openvm_cuda_common::d_buffer::DeviceBufferView;

    use super::*;

    extern "C" {
        fn _bitwise_op_lookup_tracegen(
            d_count: *const u32,
            d_cpu_count: *const u32,
            d_trace: *mut F,
            num_bits: u32,
        ) -> i32;

        #[cfg(test)]
        fn _bitwise_dummy_tracegen(
            d_trace: *mut F,
            records: DeviceBufferView,
            bitwise_count: *mut u32,
            bitwise_num_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_count: &DeviceBuffer<F>,
        d_cpu_count: &Option<DeviceBuffer<u32>>,
        d_trace: &DeviceBuffer<F>,
        num_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_bitwise_op_lookup_tracegen(
            d_count.as_ptr() as *const u32,
            d_cpu_count
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            d_trace.as_mut_ptr(),
            num_bits,
        ))
    }

    #[cfg(test)]
    pub unsafe fn dummy_tracegen(
        d_trace: &DeviceBuffer<F>,
        records: &DeviceBuffer<u32>,
        bitwise_count: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_bitwise_dummy_tracegen(
            d_trace.as_mut_ptr(),
            records.view(),
            bitwise_count.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
        ))
    }
}

pub mod range_tuple {
    use super::*;

    extern "C" {
        fn _range_tuple_checker_tracegen(
            d_count: *const u32,
            d_cpu_count: *const u32,
            d_trace: *mut F,
            num_bins: usize,
        ) -> i32;

        #[cfg(test)]
        fn _range_tuple_dummy_tracegen(
            d_data: *const u32,
            d_trace: *mut F,
            d_rc_count: *mut u32,
            data_height: usize,
            sizes: *const u32,
            num_sizes: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_count: &DeviceBuffer<F>,
        d_cpu_count: &Option<DeviceBuffer<u32>>,
        d_trace: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_tuple_checker_tracegen(
            d_count.as_ptr() as *const u32,
            d_cpu_count
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            d_trace.as_mut_ptr(),
            d_count.len(),
        ))
    }

    #[cfg(test)]
    pub unsafe fn dummy_tracegen(
        d_data: &DeviceBuffer<u32>,
        d_trace: &DeviceBuffer<F>,
        d_rc_count: &DeviceBuffer<F>,
        sizes: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_tuple_dummy_tracegen(
            d_data.as_ptr(),
            d_trace.as_mut_ptr(),
            d_rc_count.as_mut_ptr() as *mut u32,
            d_data.len() / sizes.len(),
            sizes.as_ptr(),
            sizes.len(),
        ))
    }
}

pub mod var_range {
    use super::*;

    extern "C" {
        fn _range_checker_tracegen(
            d_count: *const u32,
            d_cpu_count: *const u32,
            d_trace: *mut F,
            num_bins: usize,
        ) -> i32;

        #[cfg(test)]
        fn _var_range_dummy_tracegen(
            d_data: *const u32,
            d_trace: *mut F,
            d_rc_count: *mut u32,
            data_len: usize,
            range_max_bits: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_count: &DeviceBuffer<F>,
        d_cpu_count: &Option<DeviceBuffer<u32>>,
        d_trace: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_checker_tracegen(
            d_count.as_ptr() as *const u32,
            d_cpu_count
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            d_trace.as_mut_ptr(),
            d_count.len(),
        ))
    }

    #[cfg(test)]
    pub unsafe fn dummy_tracegen(
        d_data: &DeviceBuffer<u32>,
        d_trace: &DeviceBuffer<F>,
        d_rc_count: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_var_range_dummy_tracegen(
            d_data.as_ptr(),
            d_trace.as_mut_ptr(),
            d_rc_count.as_mut_ptr() as *mut u32,
            d_data.len(),
            d_rc_count.len(),
        ))
    }
}

#[cfg(test)]
pub mod encoder {
    use super::*;

    extern "C" {
        fn _encoder_tracegen(
            trace: *mut F,
            num_flags: u32,
            max_degree: u32,
            reserve_invalid: bool,
            expected_k: u32,
        ) -> i32;
    }

    pub unsafe fn dummy_tracegen(
        d_trace: &DeviceBuffer<F>,
        num_flags: u32,
        max_degree: u32,
        reserve_invalid: bool,
        expected_k: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_encoder_tracegen(
            d_trace.as_mut_ptr(),
            num_flags,
            max_degree,
            reserve_invalid,
            expected_k,
        ))
    }
}

#[cfg(test)]
pub mod is_equal {
    use super::*;

    extern "C" {
        fn _isequal_tracegen(output: *mut F, inputs_x: *mut F, inputs_y: *mut F, n: u32) -> i32;

        fn _isequal_array_tracegen(
            output: *mut F,
            inputs_x: *mut F,
            inputs_y: *mut F,
            array_len: u32,
            n: u32,
        ) -> i32;
    }

    pub unsafe fn dummy_tracegen(
        d_output: &DeviceBuffer<F>,
        d_inputs_x: &DeviceBuffer<F>,
        d_inputs_y: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_isequal_tracegen(
            d_output.as_mut_ptr(),
            d_inputs_x.as_mut_ptr(),
            d_inputs_y.as_mut_ptr(),
            d_inputs_x.len() as u32,
        ))
    }

    pub unsafe fn dummy_tracegen_array(
        d_output: &DeviceBuffer<F>,
        d_inputs_x: &DeviceBuffer<F>,
        d_inputs_y: &DeviceBuffer<F>,
        array_len: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_isequal_array_tracegen(
            d_output.as_mut_ptr(),
            d_inputs_x.as_mut_ptr(),
            d_inputs_y.as_mut_ptr(),
            array_len as u32,
            (d_inputs_x.len() / array_len) as u32,
        ))
    }
}

#[cfg(test)]
pub mod is_zero {
    use super::*;

    extern "C" {
        fn _iszero_tracegen(output: *mut F, inputs: *mut F, n: u32) -> i32;
    }

    pub unsafe fn dummy_tracegen(
        d_output: &DeviceBuffer<F>,
        d_inputs: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_iszero_tracegen(
            d_output.as_mut_ptr(),
            d_inputs.as_mut_ptr(),
            d_inputs.len() as u32,
        ))
    }
}

#[cfg(test)]
pub mod less_than {
    use super::*;

    extern "C" {
        fn _assert_less_than_tracegen(
            trace: *mut F,
            trace_height: usize,
            pairs: *const u32,
            max_bits: u32,
            aux_len: u32,
            rc_count: *mut u32,
            rc_num_bins: u32,
        ) -> i32;

        fn _less_than_tracegen(
            trace: *mut F,
            trace_height: usize,
            pairs: *const u32,
            max_bits: u32,
            aux_len: u32,
            rc_count: *mut u32,
            rc_num_bins: u32,
        ) -> i32;

        fn _less_than_array_tracegen(
            trace: *mut F,
            trace_height: usize,
            pairs: *const u32,
            max_bits: u32,
            array_len: u32,
            aux_len: u32,
            rc_count: *mut u32,
            rc_num_bins: u32,
        ) -> i32;
    }

    pub unsafe fn assert_less_than_dummy_tracegen(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_assert_less_than_tracegen(
            trace.as_mut_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits as u32,
            aux_len as u32,
            rc_count.as_mut_ptr(),
            rc_count.len() as u32,
        ))
    }

    pub unsafe fn less_than_dummy_tracegen(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_less_than_tracegen(
            trace.as_mut_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits as u32,
            aux_len as u32,
            rc_count.as_mut_ptr(),
            rc_count.len() as u32,
        ))
    }

    pub unsafe fn less_than_array_dummy_tracegen(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        array_len: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_less_than_array_tracegen(
            trace.as_mut_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits as u32,
            array_len as u32,
            aux_len as u32,
            rc_count.as_mut_ptr(),
            rc_count.len() as u32,
        ))
    }
}
