use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
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
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _xorin_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_error: *mut u32,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
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
            pointer_max_bits,
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
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_error: *mut u32,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_xorin_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_error,
            expected_opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
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
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _keccakf_op_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_seeds: DeviceBufferView,
            d_predecessors: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_preimages: *mut u64,
            preimage_words: usize,
            d_error: *mut u32,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
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
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_op_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            pointer_max_bits,
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
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_seeds: DeviceBufferView,
        d_predecessors: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_preimages: &DeviceBuffer<u64>,
        d_error: *mut u32,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_op_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_preimages.as_mut_ptr(),
            d_preimages.len(),
            d_error,
            expected_opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
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
            d_round_states: *mut u64,
            round_state_words: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _keccakf_perm_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_program: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            d_preimages: *const u64,
            preimage_words: usize,
            d_round_states: *mut u64,
            round_state_words: usize,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// # Safety
    /// All device buffers must be valid and properly allocated.
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        d_round_states: &DeviceBuffer<u64>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_perm_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            num_records,
            d_round_states.as_mut_ptr(),
            d_round_states.len(),
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_program: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        d_preimages: &DeviceBuffer<u64>,
        d_round_states: &DeviceBuffer<u64>,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_keccakf_perm_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_program,
            d_steps,
            step_start,
            num_steps,
            d_preimages.as_ptr(),
            d_preimages.len(),
            d_round_states.as_mut_ptr(),
            d_round_states.len(),
            d_error,
            stream,
        ))
    }
}
