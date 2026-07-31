#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

pub mod sha256 {
    use super::*;

    extern "C" {
        fn launch_sha256_second_pass_dependencies(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha256_fill_invalid_rows(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
            d_prev_hashes: *const u32,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha256_main_replay_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            ptr_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha256_block_replay_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            ptr_max_bits: u32,
            d_prev_hashes: *mut u32,
            d_bitwise_lookup: *mut u32,
            d_scratch: *mut u32,
            scratch_words: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn sha256_second_pass_dependencies(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let result =
            launch_sha256_second_pass_dependencies(d_trace.as_mut_ptr(), height, rows_used, stream);
        CudaError::from_result(result)
    }

    pub unsafe fn sha256_fill_invalid_rows(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
        d_prev_hashes: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let result = launch_sha256_fill_invalid_rows(
            d_trace.as_mut_ptr(),
            height,
            rows_used,
            d_prev_hashes.as_ptr(),
            stream,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sha256_main_replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        ptr_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(launch_sha256_main_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            ptr_max_bits,
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_error,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sha256_block_replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        ptr_max_bits: u32,
        d_prev_hashes: &DeviceBuffer<u32>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_scratch: &DeviceBuffer<u32>,
        d_range_checker: &DeviceBuffer<F>,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(launch_sha256_block_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            ptr_max_bits,
            d_prev_hashes.as_mut_ptr(),
            d_bitwise_lookup.as_mut_ptr().cast(),
            d_scratch.as_mut_ptr(),
            d_scratch.len(),
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            d_error,
            stream,
        ))
    }
}

pub mod sha512 {
    use super::*;

    extern "C" {
        fn launch_sha512_second_pass_dependencies(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha512_fill_invalid_rows(
            d_trace: *mut F,
            trace_height: usize,
            rows_used: usize,
            d_prev_hashes: *const u64,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha512_main_replay_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            ptr_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

        fn launch_sha512_block_replay_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            ptr_max_bits: u32,
            d_prev_hashes: *mut u64,
            d_bitwise_lookup: *mut u32,
            d_scratch: *mut u64,
            scratch_words: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn sha512_second_pass_dependencies(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let result =
            launch_sha512_second_pass_dependencies(d_trace.as_mut_ptr(), height, rows_used, stream);
        CudaError::from_result(result)
    }

    pub unsafe fn sha512_fill_invalid_rows(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
        d_prev_hashes: &DeviceBuffer<u64>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let result = launch_sha512_fill_invalid_rows(
            d_trace.as_mut_ptr(),
            height,
            rows_used,
            d_prev_hashes.as_ptr(),
            stream,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sha512_main_replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        ptr_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(launch_sha512_main_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            ptr_max_bits,
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_error,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sha512_block_replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        ptr_max_bits: u32,
        d_prev_hashes: &DeviceBuffer<u64>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_scratch: &DeviceBuffer<u64>,
        d_range_checker: &DeviceBuffer<F>,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(launch_sha512_block_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            ptr_max_bits,
            d_prev_hashes.as_mut_ptr(),
            d_bitwise_lookup.as_mut_ptr().cast(),
            d_scratch.as_mut_ptr(),
            d_scratch.len(),
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            d_error,
            stream,
        ))
    }
}
