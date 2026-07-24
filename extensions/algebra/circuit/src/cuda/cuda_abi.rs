#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
    stream::cudaStream_t,
};
use openvm_riscv_adapters::VecHeapTraceInput;

macro_rules! declare_replay_launcher {
    ($name:ident) => {
        unsafe extern "C" {
            fn $name(
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
                opcode_base: u32,
                register_as: u32,
                memory_as: u32,
                d_modulus_limbs: *const u16,
                d_range_checker: *mut u32,
                range_checker_bins: usize,
                pointer_max_bits: u32,
                timestamp_max_bits: u32,
                stream: cudaStream_t,
            ) -> i32;
        }
    };
}

declare_replay_launcher!(_modular_is_eq_replay_tracegen_l4);
declare_replay_launcher!(_modular_is_eq_replay_tracegen_l6);

unsafe extern "C" {
    fn _algebra_merge_range_counts(
        destination: *mut u32,
        source: *const u32,
        len: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _vec_heap_replay_gather(
        output: *mut std::ffi::c_void,
        output_len: usize,
        output_start: usize,
        num_reads: usize,
        blocks: usize,
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
        local_opcode: u32,
        register_as: u32,
        memory_as: u32,
        pointer_max_bits: u32,
        d_error: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
}

pub unsafe fn merge_range_counts(
    destination: &DeviceBuffer<F>,
    source: &DeviceBuffer<F>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert_eq!(destination.len(), source.len());
    CudaError::from_result(_algebra_merge_range_counts(
        destination.as_mut_ptr().cast(),
        source.as_ptr().cast(),
        source.len(),
        stream,
    ))
}

pub unsafe fn gather_vec_heap<const NUM_READS: usize, const BLOCKS: usize>(
    output: &DeviceBuffer<VecHeapTraceInput<NUM_READS, BLOCKS>>,
    output_start: usize,
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
    local_opcode: u32,
    register_as: u32,
    memory_as: u32,
    pointer_max_bits: u32,
    d_error: *mut u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_vec_heap_replay_gather(
        output.as_mut_ptr().cast(),
        output.len(),
        output_start,
        NUM_READS,
        BLOCKS,
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
        local_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_error,
        stream,
    ))
}

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
    opcode_base: u32,
    register_as: u32,
    memory_as: u32,
    d_modulus_limbs: &DeviceBuffer<u16>,
    d_range_checker: &DeviceBuffer<F>,
    num_lanes: usize,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let launcher = match num_lanes {
        4 => _modular_is_eq_replay_tracegen_l4,
        6 => _modular_is_eq_replay_tracegen_l6,
        _ => panic!("unsupported ModularIsEqual num_lanes {num_lanes}"),
    };
    CudaError::from_result(launcher(
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
        opcode_base,
        register_as,
        memory_as,
        d_modulus_limbs.as_ptr(),
        d_range_checker.as_mut_ptr() as *mut u32,
        d_range_checker.len(),
        pointer_max_bits,
        timestamp_max_bits,
        stream,
    ))
}
