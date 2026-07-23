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
        fn _rv64_less_than_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            slt_step_start: usize,
            num_slt_steps: usize,
            sltu_step_start: usize,
            num_sltu_steps: usize,
            d_error: *mut u32,
            slt_opcode: u32,
            sltu_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        slt_step_start: usize,
        num_slt_steps: usize,
        sltu_step_start: usize,
        num_sltu_steps: usize,
        d_error: *mut u32,
        slt_opcode: u32,
        sltu_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_rv64_less_than_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            slt_step_start,
            num_slt_steps,
            sltu_step_start,
            num_sltu_steps,
            d_error,
            slt_opcode,
            sltu_opcode,
            register_address_space,
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
        fn _rv64_shift_logical_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            sll_step_start: usize,
            num_sll_steps: usize,
            srl_step_start: usize,
            num_srl_steps: usize,
            d_error: *mut u32,
            sll_opcode: u32,
            srl_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        sll_step_start: usize,
        num_sll_steps: usize,
        srl_step_start: usize,
        num_srl_steps: usize,
        d_error: *mut u32,
        sll_opcode: u32,
        srl_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_rv64_shift_logical_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            sll_step_start,
            num_sll_steps,
            srl_step_start,
            num_srl_steps,
            d_error,
            sll_opcode,
            srl_opcode,
            register_address_space,
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
        fn _rv64_shift_right_arithmetic_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_rv64_shift_right_arithmetic_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            opcode,
            register_address_space,
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
        fn _add_sub_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            add_step_start: usize,
            num_add_steps: usize,
            sub_step_start: usize,
            num_sub_steps: usize,
            d_error: *mut u32,
            add_opcode: u32,
            sub_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        add_step_start: usize,
        num_add_steps: usize,
        sub_step_start: usize,
        num_sub_steps: usize,
        d_error: *mut u32,
        add_opcode: u32,
        sub_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_add_sub_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            add_step_start,
            num_add_steps,
            sub_step_start,
            num_sub_steps,
            d_error,
            add_opcode,
            sub_opcode,
            register_address_space,
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

        fn _addi_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            addi_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        addi_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_addi_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            addi_opcode,
            register_address_space,
            immediate_address_space,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod addi_w_cuda {
    use super::*;
    extern "C" {
        fn _addi_w_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _addi_w_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            addiw_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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
        CudaError::from_result(_addi_w_tracegen(
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        addiw_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_addi_w_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            addiw_opcode,
            register_address_space,
            immediate_address_space,
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
        fn _bitwise_logic_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            xor_step_start: usize,
            num_xor_steps: usize,
            or_step_start: usize,
            num_or_steps: usize,
            and_step_start: usize,
            num_and_steps: usize,
            d_error: *mut u32,
            xor_opcode: u32,
            or_opcode: u32,
            and_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        xor_step_start: usize,
        num_xor_steps: usize,
        or_step_start: usize,
        num_or_steps: usize,
        and_step_start: usize,
        num_and_steps: usize,
        d_error: *mut u32,
        xor_opcode: u32,
        or_opcode: u32,
        and_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_bitwise_logic_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            xor_step_start,
            num_xor_steps,
            or_step_start,
            num_or_steps,
            and_step_start,
            num_and_steps,
            d_error,
            xor_opcode,
            or_opcode,
            and_opcode,
            register_address_space,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
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

        fn _beq_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            beq_step_start: usize,
            num_beq_steps: usize,
            bne_step_start: usize,
            num_bne_steps: usize,
            d_error: *mut u32,
            beq_opcode: u32,
            bne_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        beq_step_start: usize,
        num_beq_steps: usize,
        bne_step_start: usize,
        num_bne_steps: usize,
        d_error: *mut u32,
        beq_opcode: u32,
        bne_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_beq_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            beq_step_start,
            num_beq_steps,
            bne_step_start,
            num_bne_steps,
            d_error,
            beq_opcode,
            bne_opcode,
            register_address_space,
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
        fn _add_sub_w_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            addw_step_start: usize,
            num_addw_steps: usize,
            subw_step_start: usize,
            num_subw_steps: usize,
            d_error: *mut u32,
            addw_opcode: u32,
            subw_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        addw_step_start: usize,
        num_addw_steps: usize,
        subw_step_start: usize,
        num_subw_steps: usize,
        d_error: *mut u32,
        addw_opcode: u32,
        subw_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_add_sub_w_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            addw_step_start,
            num_addw_steps,
            subw_step_start,
            num_subw_steps,
            d_error,
            addw_opcode,
            subw_opcode,
            register_address_space,
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
        fn _rv64_shift_w_logical_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            sllw_step_start: usize,
            num_sllw_steps: usize,
            srlw_step_start: usize,
            num_srlw_steps: usize,
            d_error: *mut u32,
            sllw_opcode: u32,
            srlw_opcode: u32,
            register_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen_logical(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        sllw_step_start: usize,
        num_sllw_steps: usize,
        srlw_step_start: usize,
        num_srlw_steps: usize,
        d_error: *mut u32,
        sllw_opcode: u32,
        srlw_opcode: u32,
        register_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_rv64_shift_w_logical_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            sllw_step_start,
            num_sllw_steps,
            srlw_step_start,
            num_srlw_steps,
            d_error,
            sllw_opcode,
            srlw_opcode,
            register_address_space,
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
        fn _shift_logical_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            slli_step_start: usize,
            num_slli_steps: usize,
            srli_step_start: usize,
            num_srli_steps: usize,
            d_error: *mut u32,
            slli_opcode: u32,
            srli_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        slli_step_start: usize,
        num_slli_steps: usize,
        srli_step_start: usize,
        num_srli_steps: usize,
        d_error: *mut u32,
        slli_opcode: u32,
        srli_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_shift_logical_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            slli_step_start,
            num_slli_steps,
            srli_step_start,
            num_srli_steps,
            d_error,
            slli_opcode,
            srli_opcode,
            register_address_space,
            immediate_address_space,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_w_logical_imm_cuda {
    use super::*;
    extern "C" {
        fn _shift_w_logical_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _shift_w_logical_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            slliw_step_start: usize,
            num_slliw_steps: usize,
            srliw_step_start: usize,
            num_srliw_steps: usize,
            d_error: *mut u32,
            slliw_opcode: u32,
            srliw_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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
        CudaError::from_result(_shift_w_logical_imm_tracegen(
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        slliw_step_start: usize,
        num_slliw_steps: usize,
        srliw_step_start: usize,
        num_srliw_steps: usize,
        d_error: *mut u32,
        slliw_opcode: u32,
        srliw_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_shift_w_logical_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            slliw_step_start,
            num_slliw_steps,
            srliw_step_start,
            num_srliw_steps,
            d_error,
            slliw_opcode,
            srliw_opcode,
            register_address_space,
            immediate_address_space,
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
        fn _shift_right_arithmetic_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_shift_right_arithmetic_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            opcode,
            register_address_space,
            immediate_address_space,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift_w_right_arithmetic_imm_cuda {
    use super::*;
    extern "C" {
        fn _shift_w_right_arithmetic_imm_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _shift_w_right_arithmetic_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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
        CudaError::from_result(_shift_w_right_arithmetic_imm_tracegen(
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_shift_w_right_arithmetic_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            opcode,
            register_address_space,
            immediate_address_space,
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
        fn _less_than_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            slti_step_start: usize,
            num_slti_steps: usize,
            sltiu_step_start: usize,
            num_sltiu_steps: usize,
            d_error: *mut u32,
            slti_opcode: u32,
            sltiu_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        slti_step_start: usize,
        num_slti_steps: usize,
        sltiu_step_start: usize,
        num_sltiu_steps: usize,
        d_error: *mut u32,
        slti_opcode: u32,
        sltiu_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_less_than_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            slti_step_start,
            num_slti_steps,
            sltiu_step_start,
            num_sltiu_steps,
            d_error,
            slti_opcode,
            sltiu_opcode,
            register_address_space,
            immediate_address_space,
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
        fn _bitwise_logic_imm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program_log: DeviceBufferView,
            d_memory_log: DeviceBufferView,
            d_initial_write_log: DeviceBufferView,
            d_memory_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            xori_step_start: usize,
            num_xori_steps: usize,
            ori_step_start: usize,
            num_ori_steps: usize,
            andi_step_start: usize,
            num_andi_steps: usize,
            d_error: *mut u32,
            xori_opcode: u32,
            ori_opcode: u32,
            andi_opcode: u32,
            register_address_space: u32,
            immediate_address_space: u32,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program_log: DeviceBufferView,
        d_memory_log: DeviceBufferView,
        d_initial_write_log: DeviceBufferView,
        d_memory_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        xori_step_start: usize,
        num_xori_steps: usize,
        ori_step_start: usize,
        num_ori_steps: usize,
        andi_step_start: usize,
        num_andi_steps: usize,
        d_error: *mut u32,
        xori_opcode: u32,
        ori_opcode: u32,
        andi_opcode: u32,
        register_address_space: u32,
        immediate_address_space: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two());
        CudaError::from_result(_bitwise_logic_imm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program_log,
            d_memory_log,
            d_initial_write_log,
            d_memory_predecessors,
            d_steps,
            xori_step_start,
            num_xori_steps,
            ori_step_start,
            num_ori_steps,
            andi_step_start,
            num_andi_steps,
            d_error,
            xori_opcode,
            ori_opcode,
            andi_opcode,
            register_address_space,
            immediate_address_space,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
    }
}
