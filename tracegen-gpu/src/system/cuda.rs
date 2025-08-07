#![allow(clippy::missing_safety_doc)]

use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod boundary {
    use super::*;

    extern "C" {
        fn _persistent_boundary_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_initial_mem: *const *const std::ffi::c_void,
            d_raw_records: *const u32,
            num_records: usize,
            d_poseidon2_raw_buffer: *mut std::ffi::c_void,
            d_poseidon2_buffer_idx: *mut u32,
            poseidon2_capacity: usize,
            sbox_regs: usize,
        ) -> i32;

        fn _volatile_boundary_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_raw_records: *const u32,
            num_records: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: usize,
            as_max_bits: usize,
            ptr_max_bits: usize,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn persistent_boundary_tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_initial_mem: &DeviceBuffer<*const std::ffi::c_void>,
        d_touched_blocks: &DeviceBuffer<u32>,
        num_records: usize,
        d_poseidon2_raw_buffer: &DeviceBuffer<T>,
        d_poseidon2_buffer_idx: &DeviceBuffer<u32>,
        sbox_regs: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_persistent_boundary_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_initial_mem.as_ptr(),
            d_touched_blocks.as_ptr(),
            num_records,
            d_poseidon2_raw_buffer.as_mut_raw_ptr(),
            d_poseidon2_buffer_idx.as_mut_ptr(),
            d_poseidon2_raw_buffer.len(),
            sbox_regs,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn volatile_boundary_tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u32>,
        num_records: usize,
        d_range_checker: &DeviceBuffer<T>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_volatile_boundary_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            as_max_bits,
            ptr_max_bits,
        ))
    }
}

pub mod phantom {
    use super::*;

    extern "C" {
        fn _phantom_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            num_records: usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_phantom_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
        ))
    }
}

pub mod poseidon2 {
    use super::*;

    extern "C" {
        fn _system_poseidon2_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *mut std::ffi::c_void,
            d_counts: *mut u32,
            num_records: usize,
            sbox_regs: usize,
        ) -> i32;

        fn _system_poseidon2_deduplicate_records(
            d_records: *mut std::ffi::c_void,
            d_counts: *mut u32,
            num_records: *mut usize,
        ) -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<T>,
        d_counts: &DeviceBuffer<u32>,
        num_records: usize,
        sbox_regs: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_mut_raw_ptr(),
            d_counts.as_mut_ptr(),
            num_records,
            sbox_regs,
        ))
    }

    pub unsafe fn deduplicate_records<T>(
        d_records: &DeviceBuffer<T>,
        d_counts: &DeviceBuffer<u32>,
        num_records: &mut usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_deduplicate_records(
            d_records.as_mut_raw_ptr(),
            d_counts.as_mut_ptr(),
            num_records as *mut usize,
        ))
    }
}

pub mod program {
    use super::*;

    extern "C" {
        fn _program_cached_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const std::ffi::c_void,
            num_records: usize,
            pc_base: u32,
            pc_step: u32,
            terminate_opcode: usize,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cached_tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<T>,
        num_records: usize,
        pc_base: u32,
        pc_step: u32,
        terminate_opcode: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_program_cached_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_raw_ptr(),
            num_records,
            pc_base,
            pc_step,
            terminate_opcode,
        ))
    }
}

pub mod public_values {
    use super::*;

    extern "C" {
        fn _public_values_tracegen(
            d_trace: *mut std::ffi::c_void,
            height: usize,
            width: usize,
            d_records: *const u8,
            num_records: usize,
            d_range_checker: *mut u32,
            range_checker_bins: u32,
            timestamp_max_bits: u32,
            num_custom_pvs: u32,
            max_degree: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen<T>(
        d_trace: &DeviceBuffer<T>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        d_range_checker: &DeviceBuffer<T>,
        timestamp_max_bits: u32,
        num_custom_pvs: usize,
        max_degree: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_public_values_tracegen(
            d_trace.as_mut_raw_ptr(),
            height,
            width,
            d_records.as_ptr(),
            num_records,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            num_custom_pvs as u32,
            max_degree,
        ))
    }
}

pub mod access_adapters {
    use super::*;
    use crate::system::access_adapters::{OffsetInfo, NUM_ADAPTERS};

    extern "C" {
        fn _access_adapters_tracegen(
            d_traces: *const *mut std::ffi::c_void,
            num_adapters: usize,
            d_unpadded_heights: *const usize,
            d_widths: *const usize,
            num_records: usize,
            d_records: *const u8,
            d_record_offsets: *mut u32,
            d_range_checker: *mut u32,
            range_checker_bins: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen<T>(
        d_trace_ptrs: &DeviceBuffer<*mut std::ffi::c_void>,
        d_unpadded_heights: &DeviceBuffer<usize>,
        d_widths: &DeviceBuffer<usize>,
        num_records: usize,
        d_records: &DeviceBuffer<u8>,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        d_range_checker: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_access_adapters_tracegen(
            d_trace_ptrs.as_ptr(),
            NUM_ADAPTERS,
            d_unpadded_heights.as_ptr(),
            d_widths.as_ptr(),
            num_records,
            d_records.as_ptr(),
            d_record_offsets.as_mut_ptr() as *mut u32,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
        ))
    }
}
