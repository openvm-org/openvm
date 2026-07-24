#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

pub mod add_sub256 {
    use super::*;

    extern "C" {
        fn _add_sub256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_add_sub256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod bitwise_logic256 {
    use super::*;

    extern "C" {
        fn _bitwise_logic256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            pointer_max_bits: u32,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_bitwise_logic256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod beq256 {
    use super::*;

    extern "C" {
        fn _branch_equal256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_branch_equal256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod lt256 {
    use super::*;

    extern "C" {
        fn _less_than256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_less_than256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod blt256 {
    use super::*;

    extern "C" {
        fn _branch_less_than256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_branch_less_than256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod shift256 {
    use super::*;

    extern "C" {
        fn _shift256_logical_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _shift256_right_arithmetic_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen_logical(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_shift256_logical_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }

    pub unsafe fn tracegen_right_arithmetic(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_shift256_right_arithmetic_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub mod mul256 {
    use openvm_circuit_primitives::cuda_abi::UInt2;

    use super::*;

    extern "C" {
        fn _multiplication256_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *const u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *const u32,
            d_range_tuple: *const u32,
            range_tuple_sizes: UInt2,
            pointer_max_bits: u32,
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
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_multiplication256_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple.as_mut_ptr() as *mut u32,
            range_tuple_sizes,
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

#[cfg(feature = "rvr")]
pub mod replay {
    use openvm_circuit_primitives::cuda_abi::UInt2;

    use super::*;

    #[repr(u32)]
    #[derive(Clone, Copy)]
    pub enum U16Kind {
        LessThan = 0,
        ShiftLogical = 1,
        ShiftRightArithmetic = 2,
        BranchEqual = 3,
        BranchLessThan = 4,
    }

    extern "C" {
        fn _add_sub256_replay_tracegen(
            trace: *mut F,
            height: usize,
            width: usize,
            instructions: DeviceBufferView,
            pc_base: u32,
            program_log: DeviceBufferView,
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            predecessors: DeviceBufferView,
            steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            error: *mut u32,
            opcode_base: u32,
            register_address_space: u32,
            memory_address_space: u32,
            range_checker: *mut u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _bitwise_logic256_replay_tracegen(
            trace: *mut F,
            height: usize,
            width: usize,
            instructions: DeviceBufferView,
            pc_base: u32,
            program_log: DeviceBufferView,
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            predecessors: DeviceBufferView,
            steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            error: *mut u32,
            opcode_base: u32,
            register_address_space: u32,
            memory_address_space: u32,
            range_checker: *mut u32,
            range_checker_bins: usize,
            bitwise_lookup: *mut u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _int256_u16_replay_tracegen(
            trace: *mut F,
            height: usize,
            width: usize,
            instructions: DeviceBufferView,
            pc_base: u32,
            program_log: DeviceBufferView,
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            predecessors: DeviceBufferView,
            steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            error: *mut u32,
            opcode_base: u32,
            register_address_space: u32,
            memory_address_space: u32,
            range_checker: *mut u32,
            range_checker_bins: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            kind: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _multiplication256_replay_tracegen(
            trace: *mut F,
            height: usize,
            width: usize,
            instructions: DeviceBufferView,
            pc_base: u32,
            program_log: DeviceBufferView,
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            predecessors: DeviceBufferView,
            steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            error: *mut u32,
            opcode_base: u32,
            register_address_space: u32,
            memory_address_space: u32,
            range_checker: *mut u32,
            range_checker_bins: usize,
            bitwise_lookup: *mut u32,
            range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_sub(
        trace: &DeviceBuffer<F>,
        height: usize,
        instructions: DeviceBufferView,
        pc_base: u32,
        program_log: DeviceBufferView,
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        predecessors: DeviceBufferView,
        steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        error: *mut u32,
        opcode_base: u32,
        register_address_space: u32,
        memory_address_space: u32,
        range_checker: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_add_sub256_replay_tracegen(
            trace.as_mut_ptr(),
            height,
            trace.len() / height,
            instructions,
            pc_base,
            program_log,
            memory,
            seeds,
            predecessors,
            steps,
            step_start,
            num_steps,
            error,
            opcode_base,
            register_address_space,
            memory_address_space,
            range_checker.as_mut_ptr().cast(),
            range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bitwise(
        trace: &DeviceBuffer<F>,
        height: usize,
        instructions: DeviceBufferView,
        pc_base: u32,
        program_log: DeviceBufferView,
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        predecessors: DeviceBufferView,
        steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        error: *mut u32,
        opcode_base: u32,
        register_address_space: u32,
        memory_address_space: u32,
        range_checker: &DeviceBuffer<F>,
        bitwise_lookup: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_bitwise_logic256_replay_tracegen(
            trace.as_mut_ptr(),
            height,
            trace.len() / height,
            instructions,
            pc_base,
            program_log,
            memory,
            seeds,
            predecessors,
            steps,
            step_start,
            num_steps,
            error,
            opcode_base,
            register_address_space,
            memory_address_space,
            range_checker.as_mut_ptr().cast(),
            range_checker.len(),
            bitwise_lookup.as_mut_ptr().cast(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn u16(
        trace: &DeviceBuffer<F>,
        height: usize,
        instructions: DeviceBufferView,
        pc_base: u32,
        program_log: DeviceBufferView,
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        predecessors: DeviceBufferView,
        steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        error: *mut u32,
        opcode_base: u32,
        register_address_space: u32,
        memory_address_space: u32,
        range_checker: &DeviceBuffer<F>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        kind: U16Kind,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_int256_u16_replay_tracegen(
            trace.as_mut_ptr(),
            height,
            trace.len() / height,
            instructions,
            pc_base,
            program_log,
            memory,
            seeds,
            predecessors,
            steps,
            step_start,
            num_steps,
            error,
            opcode_base,
            register_address_space,
            memory_address_space,
            range_checker.as_mut_ptr().cast(),
            range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            kind as u32,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn multiplication(
        trace: &DeviceBuffer<F>,
        height: usize,
        instructions: DeviceBufferView,
        pc_base: u32,
        program_log: DeviceBufferView,
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        predecessors: DeviceBufferView,
        steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        error: *mut u32,
        opcode_base: u32,
        register_address_space: u32,
        memory_address_space: u32,
        range_checker: &DeviceBuffer<F>,
        bitwise_lookup: &DeviceBuffer<F>,
        range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_multiplication256_replay_tracegen(
            trace.as_mut_ptr(),
            height,
            trace.len() / height,
            instructions,
            pc_base,
            program_log,
            memory,
            seeds,
            predecessors,
            steps,
            step_start,
            num_steps,
            error,
            opcode_base,
            register_address_space,
            memory_address_space,
            range_checker.as_mut_ptr().cast(),
            range_checker.len(),
            bitwise_lookup.as_mut_ptr().cast(),
            range_tuple.as_mut_ptr().cast(),
            range_tuple_sizes,
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}
