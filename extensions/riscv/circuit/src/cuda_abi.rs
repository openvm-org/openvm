#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;

/// A struct that has the same memory layout as `uint2` to be used in FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UInt2 {
    pub x: u32,
    pub y: u32,
}

impl UInt2 {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

pub mod auipc_cuda {
    use super::*;

    extern "C" {
        fn _auipc_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_auipc_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod hintstore_cuda {
    use super::{super::hintstore::OffsetInfo, *};

    extern "C" {
        pub fn _hintstore_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            d_record_offsets: *const OffsetInfo,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_hintstore_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            rows_used,
            d_record_offsets.as_ptr(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod jalr_cuda {
    use super::*;

    extern "C" {
        fn _jalr_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_jalr_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod less_than_cuda {
    use super::*;

    extern "C" {
        fn _rv64_less_than_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_less_than_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_byte_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_byte_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_byte_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_halfword_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_halfword_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_halfword_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_word_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_word_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_word_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_doubleword_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_doubleword_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_doubleword_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod store_byte_cuda {
    use super::*;

    extern "C" {
        fn _rv64_store_byte_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_store_byte_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod store_halfword_cuda {
    use super::*;

    extern "C" {
        fn _rv64_store_halfword_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_store_halfword_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod store_word_cuda {
    use super::*;

    extern "C" {
        fn _rv64_store_word_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_store_word_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod store_doubleword_cuda {
    use super::*;

    extern "C" {
        fn _rv64_store_doubleword_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_store_doubleword_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_sign_extend_byte_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_sign_extend_byte_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_sign_extend_byte_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_sign_extend_halfword_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_sign_extend_halfword_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_sign_extend_halfword_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod load_sign_extend_word_cuda {
    use super::*;

    extern "C" {
        fn _rv64_load_sign_extend_word_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_load_sign_extend_word_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod mul_cuda {
    use super::*;

    extern "C" {
        fn _mul_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_mul_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_ptr() as *mut u32,
            d_range_tuple.as_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod divrem_cuda {
    use super::*;

    extern "C" {
        pub fn _rv64_div_rem_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_div_rem_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_logical_cuda {
    use super::*;

    extern "C" {
        fn _rv64_shift_logical_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_logical_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_right_arithmetic_cuda {
    use super::*;

    extern "C" {
        fn _rv64_shift_right_arithmetic_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_right_arithmetic_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod add_sub_cuda {
    use super::*;
    extern "C" {
        fn _add_sub_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_add_sub_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod addi_cuda {
    use super::*;
    extern "C" {
        fn _addi_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_addi_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod bitwise_logic_cuda {
    use super::*;
    extern "C" {
        fn _bitwise_logic_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_bitwise_logic_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod jal_lui_cuda {
    use super::*;

    extern "C" {
        fn _jal_lui_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_jal_lui_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod beq_cuda {
    use super::*;

    extern "C" {
        fn _beq_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_beq_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod branch_lt_cuda {
    use super::*;

    extern "C" {
        fn _blt_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_blt_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod mulh_cuda {
    use super::*;

    extern "C" {
        fn _mulh_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_mulh_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod add_sub_w_cuda {
    use super::*;
    extern "C" {
        fn _rv64_add_sub_w_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_add_sub_w_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_w_cuda {
    use super::*;

    extern "C" {
        fn _rv64_shift_w_logical_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _rv64_shift_w_right_arithmetic_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen_logical(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_w_logical_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }

    pub unsafe fn tracegen_right_arithmetic(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_w_right_arithmetic_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod mul_w_cuda {
    use super::*;

    extern "C" {
        fn _rv64_mul_w_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_rv64_mul_w_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple.as_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod divrem_w_cuda {
    use super::*;

    extern "C" {
        pub fn _rv64_div_rem_w_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_div_rem_w_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_logical_imm_cuda {
    use super::*;
    extern "C" {
        fn _shift_logical_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_shift_logical_imm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_right_arithmetic_imm_cuda {
    use super::*;
    extern "C" {
        fn _shift_right_arithmetic_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_shift_right_arithmetic_imm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod less_than_imm_cuda {
    use super::*;
    extern "C" {
        fn _less_than_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_less_than_imm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod bitwise_logic_imm_cuda {
    use super::*;
    extern "C" {
        fn _bitwise_logic_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_bitwise_logic_imm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}
