#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};

pub mod count {
    use super::*;

    extern "C" {
        fn _deferral_count_tracegen(
            d_trace: *mut F,
            height: usize,
            d_count: *const u32,
            num_def_circuits: usize,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_count: &DeviceBuffer<u32>,
        num_def_circuits: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_count_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_count.as_ptr(),
            num_def_circuits,
            stream,
        ))
    }
}

pub mod poseidon2 {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct DeferralPoseidon2Count {
        pub compress_mult: u32,
        pub capacity_mult: u32,
    }

    extern "C" {
        fn _deferral_poseidon2_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *mut F,
            d_counts: *mut DeferralPoseidon2Count,
            num_records: usize,
            sbox_regs: usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _deferral_poseidon2_deduplicate_records_get_temp_bytes(
            d_records: *mut F,
            d_counts: *mut DeferralPoseidon2Count,
            num_records: usize,
            d_num_records: *mut usize,
            h_temp_bytes_out: *mut usize,
            stream: cudaStream_t,
        ) -> i32;

        fn _deferral_poseidon2_deduplicate_records(
            d_records: *mut F,
            d_counts: *mut DeferralPoseidon2Count,
            d_records_out: *mut F,
            d_counts_out: *mut DeferralPoseidon2Count,
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
        d_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        num_records: usize,
        sbox_regs: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_poseidon2_tracegen(
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
        d_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        num_records: usize,
        d_num_records: &DeviceBuffer<usize>,
        h_temp_bytes_out: &mut usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_poseidon2_deduplicate_records_get_temp_bytes(
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
        d_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        d_records_out: &DeviceBuffer<F>,
        d_counts_out: &DeviceBuffer<DeferralPoseidon2Count>,
        num_records: usize,
        d_num_records: &DeviceBuffer<usize>,
        d_temp_storage: &DeviceBuffer<u8>,
        temp_storage_bytes: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_poseidon2_deduplicate_records(
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

pub mod call {
    use super::*;
    use crate::cuda_abi::poseidon2::DeferralPoseidon2Count;

    #[allow(clippy::too_many_arguments)]
    extern "C" {
        fn _deferral_call_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            num_records: usize,
            d_count: *mut u32,
            num_def_circuits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_bitwise: *mut u32,
            d_poseidon2_records: *mut F,
            d_poseidon2_counts: *mut DeferralPoseidon2Count,
            d_poseidon2_idx: *mut u32,
            poseidon2_capacity: usize,
            address_bits: usize,
            stream: cudaStream_t,
        ) -> i32;

        #[cfg(feature = "rvr")]
        fn _deferral_call_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            instructions: DeviceBufferView,
            pc_base: u32,
            program: DeviceBufferView,
            memory: DeviceBufferView,
            seeds: DeviceBufferView,
            field_values: DeviceBufferView,
            field_seeds: DeviceBufferView,
            predecessors: DeviceBufferView,
            steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            deferral_as: u32,
            byte_pointer_bits: u32,
            d_count: *mut u32,
            num_def_circuits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_bitwise: *mut u32,
            d_poseidon2_records: *mut F,
            d_poseidon2_counts: *mut DeferralPoseidon2Count,
            d_poseidon2_idx: *mut u32,
            poseidon2_capacity: usize,
            address_bits: usize,
            error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        d_count: &DeviceBuffer<u32>,
        num_def_circuits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_bitwise: &DeviceBuffer<F>,
        d_poseidon2_records: &DeviceBuffer<F>,
        d_poseidon2_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        d_poseidon2_idx: &DeviceBuffer<u32>,
        poseidon2_capacity: usize,
        address_bits: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_call_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            d_count.as_mut_ptr(),
            num_def_circuits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_bitwise.as_mut_ptr() as *mut u32,
            d_poseidon2_records.as_mut_ptr(),
            d_poseidon2_counts.as_mut_ptr(),
            d_poseidon2_idx.as_mut_ptr(),
            poseidon2_capacity,
            address_bits,
            stream,
        ))
    }

    #[cfg(feature = "rvr")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        instructions: DeviceBufferView,
        pc_base: u32,
        program: DeviceBufferView,
        memory: DeviceBufferView,
        seeds: DeviceBufferView,
        field_values: DeviceBufferView,
        field_seeds: DeviceBufferView,
        predecessors: DeviceBufferView,
        steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        deferral_as: u32,
        byte_pointer_bits: u32,
        d_count: &DeviceBuffer<u32>,
        num_def_circuits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_bitwise: &DeviceBuffer<F>,
        d_poseidon2_records: &DeviceBuffer<F>,
        d_poseidon2_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        d_poseidon2_idx: &DeviceBuffer<u32>,
        poseidon2_capacity: usize,
        address_bits: usize,
        error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_call_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            field_values,
            field_seeds,
            predecessors,
            steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            deferral_as,
            byte_pointer_bits,
            d_count.as_mut_ptr(),
            num_def_circuits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_bitwise.as_mut_ptr() as *mut u32,
            d_poseidon2_records.as_mut_ptr(),
            d_poseidon2_counts.as_mut_ptr(),
            d_poseidon2_idx.as_mut_ptr(),
            poseidon2_capacity,
            address_bits,
            error,
            stream,
        ))
    }
}

pub mod output {
    use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

    use super::*;
    use crate::cuda_abi::poseidon2::DeferralPoseidon2Count;

    pub const COMMIT_NUM_BYTES: usize = DIGEST_SIZE * 4;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct DeferralOutputPerCall {
        pub output_commit: [u8; COMMIT_NUM_BYTES],
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct DeferralOutputPerRow {
        pub header_offset: u32,
        pub section_idx: u32,
        pub call_idx: u32,
        pub poseidon2_res: [F; DIGEST_SIZE],
    }

    #[cfg(feature = "rvr")]
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct DeferralOutputReplayCall {
        pub row_start: u32,
        pub num_rows: u32,
    }

    #[cfg(feature = "rvr")]
    const _: () = assert!(std::mem::size_of::<DeferralOutputReplayCall>() == 8);

    #[allow(clippy::too_many_arguments)]
    extern "C" {
        fn _deferral_output_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_raw_records: *const u8,
            d_per_call: *const DeferralOutputPerCall,
            d_per_row: *const DeferralOutputPerRow,
            num_valid: usize,
            d_count: *mut u32,
            num_def_circuits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_bitwise: *mut u32,
            address_bits: usize,
            d_poseidon2_records: *mut F,
            d_poseidon2_counts: *mut DeferralPoseidon2Count,
            d_poseidon2_idx: *mut u32,
            poseidon2_capacity: usize,
            stream: cudaStream_t,
        ) -> i32;

        #[cfg(feature = "rvr")]
        fn _deferral_output_replay_count_rows(
            d_counts: *mut u32,
            d_instructions: DeviceBufferView,
            pc_base: u32,
            d_program: DeviceBufferView,
            d_memory: DeviceBufferView,
            d_steps: DeviceBufferView,
            step_start: usize,
            num_steps: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            num_def_circuits: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

        #[cfg(feature = "rvr")]
        fn _deferral_output_replay_tracegen(
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
            d_calls: *const DeferralOutputReplayCall,
            rows_used: usize,
            expected_opcode: u32,
            register_as: u32,
            memory_as: u32,
            byte_pointer_bits: u32,
            d_count: *mut u32,
            num_def_circuits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_bitwise: *mut u32,
            address_bits: usize,
            d_poseidon2_records: *mut F,
            d_poseidon2_counts: *mut DeferralPoseidon2Count,
            d_poseidon2_idx: *mut u32,
            poseidon2_capacity: usize,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[cfg(feature = "rvr")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_count_rows(
        d_counts: &DeviceBuffer<u32>,
        d_instructions: DeviceBufferView,
        pc_base: u32,
        d_program: DeviceBufferView,
        d_memory: DeviceBufferView,
        d_steps: DeviceBufferView,
        step_start: usize,
        num_steps: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        num_def_circuits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_output_replay_count_rows(
            d_counts.as_mut_raw_ptr(),
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_steps,
            step_start,
            num_steps,
            expected_opcode,
            register_as,
            memory_as,
            num_def_circuits,
            d_error,
            stream,
        ))
    }

    #[cfg(feature = "rvr")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn replay_tracegen(
        d_trace: &DeviceBuffer<F>,
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
        d_calls: &DeviceBuffer<DeferralOutputReplayCall>,
        rows_used: usize,
        expected_opcode: u32,
        register_as: u32,
        memory_as: u32,
        byte_pointer_bits: u32,
        d_count: &DeviceBuffer<u32>,
        num_def_circuits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_bitwise: &DeviceBuffer<F>,
        address_bits: usize,
        d_poseidon2_records: &DeviceBuffer<F>,
        d_poseidon2_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        d_poseidon2_idx: &DeviceBuffer<u32>,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_output_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_instructions,
            pc_base,
            d_program,
            d_memory,
            d_seeds,
            d_predecessors,
            d_steps,
            step_start,
            num_steps,
            d_calls.as_ptr(),
            rows_used,
            expected_opcode,
            register_as,
            memory_as,
            byte_pointer_bits,
            d_count.as_mut_ptr(),
            num_def_circuits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_bitwise.as_mut_ptr() as *mut u32,
            address_bits,
            d_poseidon2_records.as_mut_ptr(),
            d_poseidon2_counts.as_mut_ptr(),
            d_poseidon2_idx.as_mut_ptr(),
            d_poseidon2_records.len(),
            d_error,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_raw_records: &DeviceBuffer<u8>,
        d_per_call: &DeviceBuffer<DeferralOutputPerCall>,
        d_per_row: &DeviceBuffer<DeferralOutputPerRow>,
        num_valid: usize,
        d_count: &DeviceBuffer<u32>,
        num_def_circuits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_bitwise: &DeviceBuffer<F>,
        address_bits: usize,
        d_poseidon2_records: &DeviceBuffer<F>,
        d_poseidon2_counts: &DeviceBuffer<DeferralPoseidon2Count>,
        d_poseidon2_idx: &DeviceBuffer<u32>,
        poseidon2_capacity: usize,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_deferral_output_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_raw_records.as_ptr(),
            d_per_call.as_ptr(),
            d_per_row.as_ptr(),
            num_valid,
            d_count.as_mut_ptr(),
            num_def_circuits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_bitwise.as_mut_ptr() as *mut u32,
            address_bits,
            d_poseidon2_records.as_mut_ptr(),
            d_poseidon2_counts.as_mut_ptr(),
            d_poseidon2_idx.as_mut_ptr(),
            poseidon2_capacity,
            stream,
        ))
    }
}
