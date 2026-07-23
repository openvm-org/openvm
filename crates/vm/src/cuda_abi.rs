#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

pub mod boundary {
    use super::*;

    extern "C" {
        fn _persistent_boundary_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_initial_mem: *const *const std::ffi::c_void,
            d_raw_records: *const u32,
            num_records: usize,
            d_poseidon2_raw_buffer: *mut F,
            d_poseidon2_buffer_idx: *mut u32,
            poseidon2_capacity: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn persistent_boundary_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_initial_mem: &DeviceBuffer<*const std::ffi::c_void>,
        d_touched_blocks: &DeviceBuffer<u32>,
        num_records: usize,
        d_poseidon2_raw_buffer: &DeviceBuffer<F>,
        d_poseidon2_buffer_idx: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_persistent_boundary_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_initial_mem.as_ptr(),
            d_touched_blocks.as_ptr(),
            num_records,
            d_poseidon2_raw_buffer.as_mut_ptr(),
            d_poseidon2_buffer_idx.as_mut_ptr(),
            // Length in F elements; the CUDA side converts to record count.
            d_poseidon2_raw_buffer.len(),
            stream,
        ))
    }
}

pub mod phantom {
    use super::*;

    extern "C" {
        fn _phantom_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_phantom_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            stream,
        ))
    }
}

pub mod rvr_postflight {
    use super::*;

    extern "C" {
        fn _rvr_memory_index_get_temp_bytes(
            num_entries: usize,
            h_temp_bytes_out: *mut usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _rvr_memory_index_sort(
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            address_spaces: DeviceBufferView,
            address_space_offset: u32,
            address_space_height: u32,
            pointer_max_bits: u32,
            keys_in: *mut u64,
            keys_out: *mut u64,
            temp_storage: *mut std::ffi::c_void,
            temp_storage_bytes: usize,
            error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _rvr_memory_index_scatter(
            memory: DeviceBufferView,
            num_seeds: usize,
            sorted_keys: *const u64,
            num_entries: usize,
            predecessors: *mut u32,
            touched_flags: *mut u32,
            touched_positions: *mut u32,
            touched_blocks: *mut std::ffi::c_void,
            num_touched_blocks: *mut u32,
            temp_storage: *mut std::ffi::c_void,
            temp_storage_bytes: usize,
            error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _rvr_program_index_get_temp_bytes(
            num_steps: usize,
            h_temp_bytes_out: *mut usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _rvr_program_index(
            instructions: DeviceBufferView,
            dense_program_rows: DeviceBufferView,
            pc_base: u32,
            program: DeviceBufferView,
            memory: DeviceBufferView,
            active_opcodes: DeviceBufferView,
            timestamp_max_bits: u32,
            endpoint_kind: u32,
            resume_pc: u32,
            final_timestamp: u32,
            terminate_opcode: u32,
            opcode_keys_in: *mut u32,
            opcode_keys_out: *mut u32,
            steps_in: *mut std::ffi::c_void,
            steps_out: *mut std::ffi::c_void,
            ranges: *mut u32,
            program_frequencies: *mut u32,
            temp_storage: *mut std::ffi::c_void,
            temp_storage_bytes: usize,
            error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn memory_index_get_temp_bytes(
        num_entries: usize,
        h_temp_bytes_out: &mut usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_memory_index_get_temp_bytes(
            num_entries,
            h_temp_bytes_out,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn memory_index_sort(
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        address_spaces: DeviceBufferView,
        address_space_offset: u32,
        address_space_height: u32,
        pointer_max_bits: u32,
        keys_in: &DeviceBuffer<u64>,
        keys_out: &DeviceBuffer<u64>,
        temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_memory_index_sort(
            memory,
            seeds,
            address_spaces,
            address_space_offset,
            address_space_height,
            pointer_max_bits,
            keys_in.as_mut_ptr(),
            keys_out.as_mut_ptr(),
            temp_storage.as_mut_raw_ptr(),
            temp_storage_bytes,
            error.as_mut_ptr(),
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn memory_index_scatter(
        memory: DeviceBufferView,
        num_seeds: usize,
        sorted_keys: &DeviceBuffer<u64>,
        num_entries: usize,
        predecessors: &DeviceBuffer<u32>,
        touched_flags: &DeviceBuffer<u32>,
        touched_positions: &DeviceBuffer<u32>,
        touched_blocks: *mut std::ffi::c_void,
        num_touched_blocks: &DeviceBuffer<u32>,
        temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_memory_index_scatter(
            memory,
            num_seeds,
            sorted_keys.as_ptr(),
            num_entries,
            predecessors.as_mut_ptr(),
            touched_flags.as_mut_ptr(),
            touched_positions.as_mut_ptr(),
            touched_blocks,
            num_touched_blocks.as_mut_ptr(),
            temp_storage.as_mut_raw_ptr(),
            temp_storage_bytes,
            error.as_mut_ptr(),
            stream,
        ))
    }

    pub unsafe fn program_index_get_temp_bytes(
        num_steps: usize,
        h_temp_bytes_out: &mut usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_program_index_get_temp_bytes(
            num_steps,
            h_temp_bytes_out,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn program_index(
        instructions: DeviceBufferView,
        dense_program_rows: DeviceBufferView,
        pc_base: u32,
        program: DeviceBufferView,
        memory: DeviceBufferView,
        active_opcodes: DeviceBufferView,
        timestamp_max_bits: u32,
        endpoint_kind: u32,
        resume_pc: u32,
        final_timestamp: u32,
        terminate_opcode: u32,
        opcode_keys_in: &DeviceBuffer<u32>,
        opcode_keys_out: &DeviceBuffer<u32>,
        steps_in: *mut std::ffi::c_void,
        steps_out: *mut std::ffi::c_void,
        ranges: &DeviceBuffer<u32>,
        program_frequencies: &DeviceBuffer<u32>,
        temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_program_index(
            instructions,
            dense_program_rows,
            pc_base,
            program,
            memory,
            active_opcodes,
            timestamp_max_bits,
            endpoint_kind,
            resume_pc,
            final_timestamp,
            terminate_opcode,
            opcode_keys_in.as_mut_ptr(),
            opcode_keys_out.as_mut_ptr(),
            steps_in,
            steps_out,
            ranges.as_mut_ptr(),
            program_frequencies.as_mut_ptr(),
            temp_storage.as_mut_raw_ptr(),
            temp_storage_bytes,
            error.as_mut_ptr(),
            stream,
        ))
    }
}

pub mod poseidon2 {
    use super::*;

    extern "C" {
        fn _system_poseidon2_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *mut F,
            d_counts: *mut u32,
            num_records: usize,
            sbox_regs: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _system_poseidon2_deduplicate_records_get_temp_bytes(
            d_records: *mut F,
            d_counts: *mut u32,
            num_records: usize,
            d_num_records: *mut usize,
            h_temp_bytes_out: *mut usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _system_poseidon2_deduplicate_records(
            d_records: *mut F,
            d_counts: *mut u32,
            d_records_out: *mut F,
            d_counts_out: *mut u32,
            num_records: usize,
            d_num_records: *mut usize,
            d_temp_storage: *mut std::ffi::c_void,
            temp_storage_bytes: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<F>,
        d_counts: &DeviceBuffer<u32>,
        num_records: usize,
        sbox_regs: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_mut_ptr(),
            d_counts.as_mut_ptr(),
            num_records,
            sbox_regs,
            stream,
        ))
    }

    pub unsafe fn deduplicate_records_get_temp_bytes(
        d_records: &DeviceBuffer<F>,
        d_counts: &DeviceBuffer<u32>,
        num_records: usize,
        d_num_records: &DeviceBuffer<usize>,
        h_temp_bytes_out: &mut usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_deduplicate_records_get_temp_bytes(
            d_records.as_mut_ptr(),
            d_counts.as_mut_ptr(),
            num_records,
            d_num_records.as_mut_ptr(),
            h_temp_bytes_out,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn deduplicate_records(
        d_records: &DeviceBuffer<F>,
        d_counts: &DeviceBuffer<u32>,
        d_records_out: &DeviceBuffer<F>,
        d_counts_out: &DeviceBuffer<u32>,
        num_records: usize,
        d_num_records: &DeviceBuffer<usize>,
        d_temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_deduplicate_records(
            d_records.as_mut_ptr(),
            d_counts.as_mut_ptr(),
            d_records_out.as_mut_ptr(),
            d_counts_out.as_mut_ptr(),
            num_records,
            d_num_records.as_mut_ptr(),
            d_temp_storage.as_mut_raw_ptr(),
            temp_storage_bytes,
            stream,
        ))
    }
}

pub mod inventory {
    use super::*;

    extern "C" {
        fn _inventory_merge_records_get_temp_bytes(
            d_flags: *mut u32,
            in_num_records: usize,
            h_temp_bytes_out: *mut usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _inventory_merge_records(
            d_in_records: *const u32,
            in_num_records: usize,
            d_initial_mem: *const *const std::ffi::c_void,
            d_tmp_records: *mut u32,
            d_out_records: *mut u32,
            d_flags: *mut u32,
            d_positions: *mut u32,
            d_temp_storage: *mut std::ffi::c_void,
            temp_storage_bytes: usize,
            out_num_records: *mut usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn merge_records(
        d_in_records: &DeviceBuffer<u32>,
        in_num_records: usize,
        d_initial_mem: &DeviceBuffer<*const std::ffi::c_void>,
        d_tmp_records: &DeviceBuffer<u32>,
        d_out_records: &DeviceBuffer<u32>,
        d_flags: &DeviceBuffer<u32>,
        d_positions: &DeviceBuffer<u32>,
        d_temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        d_out_num_records: &DeviceBuffer<usize>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_inventory_merge_records(
            d_in_records.as_ptr(),
            in_num_records,
            d_initial_mem.as_ptr(),
            d_tmp_records.as_mut_ptr(),
            d_out_records.as_mut_ptr(),
            d_flags.as_mut_ptr(),
            d_positions.as_mut_ptr(),
            d_temp_storage.as_mut_raw_ptr(),
            temp_storage_bytes,
            d_out_num_records.as_mut_ptr(),
            stream,
        ))
    }

    extern "C" {
        fn _inventory_to_merkle_records(
            d_out_records: *const u32,
            num_records: usize,
            d_merkle_records: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// Converts merged inventory records (boundary layout) into Merkle
    /// touched-block records on device. `d_merkle_records` must hold exactly
    /// `num_records * MERKLE_TOUCHED_BLOCK_WIDTH` words.
    pub unsafe fn to_merkle_records(
        d_out_records: &DeviceBuffer<u32>,
        num_records: usize,
        d_merkle_records: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_inventory_to_merkle_records(
            d_out_records.as_ptr(),
            num_records,
            d_merkle_records.as_mut_ptr(),
            stream,
        ))
    }

    pub unsafe fn merge_records_get_temp_bytes(
        d_flags: &DeviceBuffer<u32>,
        in_num_records: usize,
        h_temp_bytes_out: &mut usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_inventory_merge_records_get_temp_bytes(
            d_flags.as_mut_ptr(),
            in_num_records,
            h_temp_bytes_out,
            stream,
        ))
    }
}

pub mod program {
    use super::*;

    extern "C" {
        fn _program_fill_frequencies(
            d_freqs: *const u32,
            filtered_len: usize,
            d_out: *mut F,
            height: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// Converts raw u32 execution frequencies to field elements on device,
    /// zero-filling `[filtered_len, height)`.
    pub unsafe fn fill_frequencies(
        d_freqs: &DeviceBuffer<u32>,
        filtered_len: usize,
        d_out: &DeviceBuffer<F>,
        height: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_program_fill_frequencies(
            d_freqs.as_ptr(),
            filtered_len,
            d_out.as_mut_ptr(),
            height,
            stream,
        ))
    }

    extern "C" {
        fn _program_cached_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pc_base: u32,
            terminate_opcode: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cached_tracegen<T>(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<T>,
        pc_base: u32,
        terminate_opcode: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_program_cached_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            pc_base,
            terminate_opcode,
            stream,
        ))
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub use testing::*;

#[cfg(any(test, feature = "test-utils"))]
mod testing {
    use super::*;

    pub mod execution_testing {
        use super::*;

        unsafe extern "C" {
            unsafe fn _execution_testing_tracegen(
                d_trace: *mut F,
                height: usize,
                width: usize,
                d_records: DeviceBufferView,
                stream: cudaStream_t,
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<u8>,
            stream: cudaStream_t,
        ) -> Result<(), CudaError> {
            assert!(height.is_power_of_two());
            CudaError::from_result(_execution_testing_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                stream,
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
                stream: cudaStream_t,
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<F>,
            num_records: usize,
            block_size: usize,
            stream: cudaStream_t,
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
                stream,
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
                stream: cudaStream_t,
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<u8>,
            num_records: usize,
            stream: cudaStream_t,
        ) -> Result<(), CudaError> {
            assert!(height.is_power_of_two());
            assert!(height >= num_records);
            CudaError::from_result(_program_testing_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.as_ptr(),
                num_records,
                stream,
            ))
        }
    }
}
