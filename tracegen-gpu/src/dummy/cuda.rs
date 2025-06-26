#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod dummy_chip {
    use super::*;

    extern "C" {
        fn _dummy_tracegen(
            d_data: *const u32,
            d_trace: *mut std::ffi::c_void,
            d_rc_count: *mut u32,
            data_len: usize,
            range_max_bits: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_data: &DeviceBuffer<u32>,
        d_trace: &DeviceBuffer<T>,
        d_rc_count: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_dummy_tracegen(
            d_data.as_ptr(),
            d_trace.as_mut_raw_ptr(),
            d_rc_count.as_mut_ptr() as *mut u32,
            d_data.len(),
            d_rc_count.len(),
        ))
    }
}

pub mod encoder {
    use super::*;

    extern "C" {
        fn _encoder_tracegen(
            trace: *mut std::ffi::c_void,
            num_flags: u32,
            max_degree: u32,
            reserve_invalid: bool,
            expected_k: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        num_flags: u32,
        max_degree: u32,
        reserve_invalid: bool,
        expected_k: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_encoder_tracegen(
            d_trace.as_mut_raw_ptr(),
            num_flags,
            max_degree,
            reserve_invalid,
            expected_k,
        ))
    }
}

pub mod is_zero {
    use super::*;

    extern "C" {
        fn _iszero_tracegen(
            output: *mut std::ffi::c_void,
            inputs: *mut std::ffi::c_void,
            n: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_output: &DeviceBuffer<T>,
        d_inputs: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_iszero_tracegen(
            d_output.as_mut_raw_ptr(),
            d_inputs.as_mut_raw_ptr(),
            d_inputs.len(),
        ))
    }
}

pub mod is_equal {
    use super::*;

    extern "C" {
        fn _isequal_tracegen(
            output: *mut std::ffi::c_void,
            inputs_x: *mut std::ffi::c_void,
            inputs_y: *mut std::ffi::c_void,
            n: usize,
        ) -> i32;

        fn _isequal_array_tracegen(
            output: *mut std::ffi::c_void,
            inputs_x: *mut std::ffi::c_void,
            inputs_y: *mut std::ffi::c_void,
            array_len: usize,
            n: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_output: &DeviceBuffer<T>,
        d_inputs_x: &DeviceBuffer<T>,
        d_inputs_y: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_isequal_tracegen(
            d_output.as_mut_raw_ptr(),
            d_inputs_x.as_mut_raw_ptr(),
            d_inputs_y.as_mut_raw_ptr(),
            d_inputs_x.len(),
        ))
    }

    pub unsafe fn tracegen_array<T>(
        d_output: &DeviceBuffer<T>,
        d_inputs_x: &DeviceBuffer<T>,
        d_inputs_y: &DeviceBuffer<T>,
        array_len: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_isequal_array_tracegen(
            d_output.as_mut_raw_ptr(),
            d_inputs_x.as_mut_raw_ptr(),
            d_inputs_y.as_mut_raw_ptr(),
            array_len,
            d_inputs_x.len() / array_len,
        ))
    }
}

pub mod range_tuple_dummy {
    use super::*;

    extern "C" {
        fn _range_tuple_dummy_tracegen(
            d_data: *const u32,
            d_trace: *mut std::ffi::c_void,
            d_rc_count: *mut u32,
            data_height: usize,
            sizes: *const u32,
            num_sizes: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_data: &DeviceBuffer<u32>,
        d_trace: &DeviceBuffer<T>,
        d_rc_count: &DeviceBuffer<T>,
        sizes: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_range_tuple_dummy_tracegen(
            d_data.as_ptr(),
            d_trace.as_mut_raw_ptr(),
            d_rc_count.as_mut_ptr() as *mut u32,
            d_data.len() / sizes.len(),
            sizes.as_ptr(),
            sizes.len(),
        ))
    }
}

pub mod fibair {
    use super::*;

    extern "C" {
        fn _fibair_tracegen(output: *mut std::ffi::c_void, a: u32, b: u32, n: usize) -> i32;
    }

    pub unsafe fn fibair_tracegen<F>(
        output: &DeviceBuffer<F>,
        a: u32,
        b: u32,
        n: usize,
    ) -> Result<(), CudaError> {
        assert!(n.is_power_of_two());
        CudaError::from_result(_fibair_tracegen(output.as_mut_raw_ptr(), a, b, n))
    }
}

pub mod less_than {
    use super::*;

    extern "C" {
        fn _assert_less_than_tracegen(
            trace: *mut std::ffi::c_void,
            trace_height: usize,
            pairs: *const u32,
            max_bits: usize,
            aux_len: usize,
            rc_count: *mut u32,
            rc_num_bins: usize,
        ) -> i32;

        fn _less_than_tracegen(
            trace: *mut std::ffi::c_void,
            trace_height: usize,
            pairs: *const u32,
            max_bits: usize,
            aux_len: usize,
            rc_count: *mut u32,
            rc_num_bins: usize,
        ) -> i32;

        fn _less_than_array_tracegen(
            trace: *mut std::ffi::c_void,
            trace_height: usize,
            pairs: *const u32,
            max_bits: usize,
            array_len: usize,
            aux_len: usize,
            rc_count: *mut u32,
            rc_num_bins: usize,
        ) -> i32;
    }

    pub unsafe fn assert_less_than_tracegen<F>(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_assert_less_than_tracegen(
            trace.as_mut_raw_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits,
            aux_len,
            rc_count.as_mut_ptr(),
            rc_count.len(),
        ))
    }

    pub unsafe fn less_than_tracegen<F>(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_less_than_tracegen(
            trace.as_mut_raw_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits,
            aux_len,
            rc_count.as_mut_ptr(),
            rc_count.len(),
        ))
    }

    pub unsafe fn less_than_array_tracegen<F>(
        trace: &DeviceBuffer<F>,
        trace_height: usize,
        pairs: &DeviceBuffer<u32>,
        max_bits: usize,
        array_len: usize,
        aux_len: usize,
        rc_count: &DeviceBuffer<u32>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_less_than_array_tracegen(
            trace.as_mut_raw_ptr(),
            trace_height,
            pairs.as_ptr(),
            max_bits,
            array_len,
            aux_len,
            rc_count.as_mut_ptr(),
            rc_count.len(),
        ))
    }
}

pub mod poseidon2 {

    /// Poseidon2 tracegen on GPU (parallelized over rows)
    ///
    /// # Arguments
    ///
    /// * `d_output` - DeviceBuffer for the output (column major)
    /// * `d_inputs` - DeviceBuffer for the inputs (column major)
    /// * `sbox_regs` - Number of sbox registers (0 or 1)
    /// * `n` - Number of rows
    ///
    /// Currently only supports same constants as  
    /// https://github.com/openvm-org/openvm/blob/08bbf79368b07437271aeacb25fb8857980ca863/crates/circuits/poseidon2-air/src/lib.rs
    /// so:
    /// * `WIDTH` - 16
    /// * `SBOX_DEGREE` - 7
    /// * `HALF_FULL_ROUNDS` - 4
    /// * `PARTIAL_ROUNDS` - 13
    ///
    ///
    use super::*;

    extern "C" {
        fn _poseidon2_tracegen(
            output: *mut std::ffi::c_void,
            inputs: *mut std::ffi::c_void,
            sbox_regs: u32,
            n: u32,
        ) -> i32;

        fn _print_poseidon2_constants() -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_output: &DeviceBuffer<T>,
        d_inputs: &DeviceBuffer<T>,
        sbox_regs: u32,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_poseidon2_tracegen(
            d_output.as_mut_raw_ptr(),
            d_inputs.as_mut_raw_ptr(),
            sbox_regs,
            n,
        ))
    }
}

pub mod utils {
    use super::*;

    extern "C" {
        fn _send_bitwise_operation_lookups(
            d_count: *mut u32,
            num_bits: u32,
            pairs: *const u32,
            num_pairs: usize,
        ) -> i32;
    }

    pub unsafe fn send_bitwise_operation_lookups<T>(
        d_count: &DeviceBuffer<T>,
        pairs: &DeviceBuffer<u32>,
        num_bits: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_send_bitwise_operation_lookups(
            d_count.as_ptr() as *mut u32,
            num_bits,
            pairs.as_ptr(),
            pairs.len() / 2,
        ))
    }
}
