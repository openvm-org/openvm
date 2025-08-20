#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
};

use crate::system::cuda::access_adapters::{OffsetInfo, NUM_ADAPTERS};

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
        ) -> i32;

        fn _volatile_boundary_tracegen(
            d_trace: *mut F,
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
    pub unsafe fn persistent_boundary_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_initial_mem: &DeviceBuffer<*const std::ffi::c_void>,
        d_touched_blocks: &DeviceBuffer<u32>,
        num_records: usize,
        d_poseidon2_raw_buffer: &DeviceBuffer<F>,
        d_poseidon2_buffer_idx: &DeviceBuffer<u32>,
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
            d_poseidon2_raw_buffer.len(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn volatile_boundary_tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u32>,
        num_records: usize,
        d_range_checker: &DeviceBuffer<F>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_volatile_boundary_tracegen(
            d_trace.as_mut_ptr(),
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
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_phantom_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
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
        ) -> i32;

        fn _system_poseidon2_deduplicate_records(
            d_records: *mut F,
            d_counts: *mut u32,
            num_records: *mut usize,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<F>,
        d_counts: &DeviceBuffer<u32>,
        num_records: usize,
        sbox_regs: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.as_mut_ptr(),
            d_counts.as_mut_ptr(),
            num_records,
            sbox_regs,
        ))
    }

    pub unsafe fn deduplicate_records(
        d_records: &DeviceBuffer<F>,
        d_counts: &DeviceBuffer<u32>,
        num_records: &mut usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_system_poseidon2_deduplicate_records(
            d_records.as_mut_ptr(),
            d_counts.as_mut_ptr(),
            num_records as *mut usize,
        ))
    }
}

pub mod program {
    use super::*;

    extern "C" {
        fn _program_cached_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pc_base: u32,
            pc_step: u32,
            terminate_opcode: usize,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cached_tracegen<T>(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<T>,
        pc_base: u32,
        pc_step: u32,
        terminate_opcode: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_program_cached_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
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
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: u32,
            timestamp_max_bits: u32,
            num_custom_pvs: u32,
            max_degree: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        num_custom_pvs: usize,
        max_degree: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_public_values_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
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
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace_ptrs: &DeviceBuffer<*mut std::ffi::c_void>,
        d_unpadded_heights: &DeviceBuffer<usize>,
        d_widths: &DeviceBuffer<usize>,
        num_records: usize,
        d_records: &DeviceBuffer<u8>,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: usize,
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
            timestamp_max_bits as u32,
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
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<u8>,
        ) -> Result<(), CudaError> {
            assert!(height.is_power_of_two());
            CudaError::from_result(_execution_testing_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
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
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<F>,
            num_records: usize,
            block_size: usize,
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
            ) -> i32;
        }

        pub unsafe fn tracegen(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            width: usize,
            d_records: &DeviceBuffer<u8>,
            num_records: usize,
        ) -> Result<(), CudaError> {
            assert!(height.is_power_of_two());
            assert!(height >= num_records);
            CudaError::from_result(_program_testing_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.as_ptr(),
                num_records,
            ))
        }
    }
}
