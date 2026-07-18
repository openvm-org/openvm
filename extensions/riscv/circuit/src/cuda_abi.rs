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

#[cfg(all(feature = "cuda", feature = "rvr"))]
pub mod rvr_delta_cuda {
    use super::*;
    use crate::rvr_gpu_decode::{DeltaAirKind, DeltaAirOutputDesc, DeviceOperandEntry};

    extern "C" {
        fn _rvr_delta_predecode(
            d_delta: DeviceBufferView,
            delta_count: usize,
            d_memory: DeviceBufferView,
            memory_count: usize,
            memory_stride: usize,
            d_program: DeviceBufferView,
            program_count: usize,
            d_program_runs: *const openvm_circuit::arch::rvr::ProgramRunEntry,
            program_run_count: usize,
            program_instruction_count: usize,
            d_program_frequencies: *mut u32,
            program_frequency_count: usize,
            d_program_chronology: *mut openvm_circuit::arch::rvr::DeviceProgramEntry,
            d_initial_memory: DeviceBufferView,
            initial_memory_count: usize,
            d_touched_output: DeviceBufferView,
            d_touched_count: *mut u32,
            d_memory_prev_timestamps: *mut u32,
            d_memory_prev_values: *mut u64,
            d_operand_table: *const DeviceOperandEntry,
            operand_count: usize,
            pc_base: u32,
            d_arena_native_flags: *const u8,
            num_airs: usize,
            d_outputs: *const DeltaAirOutputDesc,
            d_expected_blocks: DeviceBufferView,
            d_expected_modes: DeviceBufferView,
            profile_segment_id: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _rvr_expand_compact_multiblock(
            d_records: DeviceBufferView,
            record_count: usize,
            d_memory: DeviceBufferView,
            memory_count: usize,
            d_operand_table: *const DeviceOperandEntry,
            operand_count: usize,
            pc_base: u32,
            kind: u32,
            d_output: DeviceBufferView,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn predecode(
        d_delta: &DeviceBuffer<u8>,
        delta_count: usize,
        d_memory: &DeviceBuffer<u8>,
        memory_count: usize,
        memory_stride: usize,
        d_program: &DeviceBuffer<openvm_circuit::arch::rvr::ProgramLogEntry>,
        program_count: usize,
        d_program_runs: &DeviceBuffer<openvm_circuit::arch::rvr::ProgramRunEntry>,
        program_instruction_count: usize,
        d_program_frequencies: &DeviceBuffer<u32>,
        d_program_chronology: &DeviceBuffer<openvm_circuit::arch::rvr::DeviceProgramEntry>,
        d_initial_memory: &DeviceBuffer<openvm_circuit::system::cuda::memory::DeviceInitialMemory>,
        d_touched_output: &DeviceBuffer<u32>,
        d_touched_count: &DeviceBuffer<u32>,
        d_memory_prev_timestamps: &DeviceBuffer<u32>,
        d_memory_prev_values: &DeviceBuffer<u64>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_arena_native_flags: &DeviceBuffer<u8>,
        d_outputs: &DeviceBuffer<DeltaAirOutputDesc>,
        d_expected_blocks: &DeviceBuffer<u64>,
        d_expected_modes: &DeviceBuffer<u8>,
        profile_segment_id: u32,
        d_error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_delta_predecode(
            d_delta.view(),
            delta_count,
            d_memory.view(),
            memory_count,
            memory_stride,
            d_program.view(),
            program_count,
            d_program_runs.as_ptr(),
            d_program_runs.len(),
            program_instruction_count,
            d_program_frequencies.as_mut_ptr(),
            d_program_frequencies.len(),
            d_program_chronology.as_mut_ptr(),
            d_initial_memory.view(),
            d_initial_memory.len(),
            d_touched_output.view(),
            d_touched_count.as_mut_ptr(),
            d_memory_prev_timestamps.as_mut_ptr(),
            d_memory_prev_values.as_mut_ptr(),
            d_operand_table.as_ptr().cast(),
            d_operand_table.len() / std::mem::size_of::<DeviceOperandEntry>(),
            pc_base,
            d_arena_native_flags.as_ptr(),
            d_arena_native_flags.len(),
            d_outputs.as_ptr(),
            d_expected_blocks.view(),
            d_expected_modes.view(),
            profile_segment_id,
            d_error.as_mut_ptr(),
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn expand_compact_multiblock(
        d_records: &DeviceBuffer<u8>,
        record_count: usize,
        d_memory: &DeviceBuffer<u8>,
        memory_count: usize,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        kind: DeltaAirKind,
        d_output: &DeviceBuffer<u8>,
        d_error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_expand_compact_multiblock(
            d_records.view(),
            record_count,
            d_memory.view(),
            memory_count,
            d_operand_table.as_ptr().cast(),
            d_operand_table.len() / std::mem::size_of::<DeviceOperandEntry>(),
            pc_base,
            kind as u32,
            d_output.view(),
            d_error.as_mut_ptr(),
            stream,
        ))
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
pub mod rvr_g2_cuda {
    use super::*;
    use crate::rvr_gpu_decode::{
        DeltaAirKind, DeltaAirOutputDesc, DeviceOperandEntry, G2ExpectedKindV1, G2ExpectedOpaqueV1,
        G2TraceSource,
    };

    extern "C" {
        fn _rvr_g2_device_pool_configure(begin: i32, reserve_bytes: usize, stats: *mut u64) -> i32;
        fn _rvr_g2_device_pool_stats(stats: *mut u64) -> i32;
        fn _rvr_g2_predecode(
            d_wire: DeviceBufferView,
            logical_wire_bytes: usize,
            run_count: usize,
            instruction_count: usize,
            d_expected_fingerprint: *const u8,
            d_blocks: *const openvm_circuit::arch::rvr::RvrG2BlockEntryV1,
            block_count: usize,
            d_operands: *const DeviceOperandEntry,
            operand_count: usize,
            pc_base: u32,
            d_initial_memory: DeviceBufferView,
            initial_timestamp: u32,
            d_expected_kinds: *const G2ExpectedKindV1,
            expected_kind_count: usize,
            d_expected_opaque: *const G2ExpectedOpaqueV1,
            expected_opaque_count: usize,
            d_program_frequencies: *mut u32,
            frequency_count: usize,
            total_record_count: usize,
            d_prepared: DeviceBufferView,
            d_row_instructions: DeviceBufferView,
            d_timestamp_offsets: DeviceBufferView,
            d_timeline: DeviceBufferView,
            d_opaque_residual_output: DeviceBufferView,
            d_opaque_residual_count: *mut u32,
            d_outputs: *const DeltaAirOutputDesc,
            num_airs: usize,
            d_touched_output: DeviceBufferView,
            d_touched_count: *mut u32,
            d_dirty_pages: DeviceBufferView,
            d_opaque_prev_timestamps: *mut u32,
            d_opaque_prev_values: *mut u64,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _rvr_g2_tracegen(
            kind: u32,
            d_trace: *mut F,
            height: usize,
            width: usize,
            source: G2TraceSource,
            d_operand_table: *const DeviceOperandEntry,
            pc_base: u32,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
        fn _rvr_g2_tracegen_reference(
            kind: u32,
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const DeviceOperandEntry,
            pc_base: u32,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn configure_device_pool(
        begin: bool,
        reserve_bytes: usize,
    ) -> Result<[u64; 8], CudaError> {
        let mut stats = [0u64; 8];
        CudaError::from_result(_rvr_g2_device_pool_configure(
            i32::from(begin),
            reserve_bytes,
            stats.as_mut_ptr(),
        ))?;
        Ok(stats)
    }

    pub unsafe fn device_pool_stats() -> Result<[u64; 5], CudaError> {
        let mut stats = [0u64; 5];
        CudaError::from_result(_rvr_g2_device_pool_stats(stats.as_mut_ptr()))?;
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn predecode(
        d_wire: &DeviceBuffer<u8>,
        logical_wire_bytes: usize,
        run_count: usize,
        instruction_count: usize,
        d_expected_fingerprint: &DeviceBuffer<u8>,
        d_blocks: &DeviceBuffer<openvm_circuit::arch::rvr::RvrG2BlockEntryV1>,
        d_operands: &DeviceBuffer<u8>,
        pc_base: u32,
        d_initial_memory: &DeviceBuffer<openvm_circuit::system::cuda::memory::DeviceInitialMemory>,
        initial_timestamp: u32,
        d_expected_kinds: &DeviceBuffer<G2ExpectedKindV1>,
        d_expected_opaque: &DeviceBuffer<G2ExpectedOpaqueV1>,
        expected_opaque_count: usize,
        d_program_frequencies: &DeviceBuffer<u32>,
        total_record_count: usize,
        d_prepared: &DeviceBuffer<u8>,
        d_row_instructions: &DeviceBuffer<u32>,
        d_timestamp_offsets: &DeviceBuffer<u32>,
        d_timeline: &DeviceBuffer<u8>,
        d_opaque_residual_output: &DeviceBuffer<u8>,
        d_opaque_residual_count: &DeviceBuffer<u32>,
        d_outputs: &DeviceBuffer<DeltaAirOutputDesc>,
        d_touched_output: &DeviceBuffer<u32>,
        d_touched_count: &DeviceBuffer<u32>,
        d_dirty_pages: &DeviceBuffer<u64>,
        d_opaque_prev_timestamps: &DeviceBuffer<u32>,
        d_opaque_prev_values: &DeviceBuffer<u64>,
        d_error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_g2_predecode(
            d_wire.view(),
            logical_wire_bytes,
            run_count,
            instruction_count,
            d_expected_fingerprint.as_ptr(),
            d_blocks.as_ptr(),
            d_blocks.len(),
            d_operands.as_ptr().cast(),
            d_operands.len() / std::mem::size_of::<DeviceOperandEntry>(),
            pc_base,
            d_initial_memory.view(),
            initial_timestamp,
            d_expected_kinds.as_ptr(),
            d_expected_kinds.len(),
            d_expected_opaque.as_ptr(),
            expected_opaque_count,
            d_program_frequencies.as_mut_ptr(),
            d_program_frequencies.len(),
            total_record_count,
            d_prepared.view(),
            d_row_instructions.view(),
            d_timestamp_offsets.view(),
            d_timeline.view(),
            d_opaque_residual_output.view(),
            d_opaque_residual_count.as_mut_ptr(),
            d_outputs.as_ptr(),
            d_outputs.len(),
            d_touched_output.view(),
            d_touched_count.as_mut_ptr(),
            d_dirty_pages.view(),
            d_opaque_prev_timestamps.as_mut_ptr(),
            d_opaque_prev_values.as_mut_ptr(),
            d_error.as_mut_ptr(),
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        kind: DeltaAirKind,
        d_trace: &DeviceBuffer<F>,
        height: usize,
        source: G2TraceSource,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: Option<&DeviceBuffer<F>>,
        d_range_tuple_checker: Option<&DeviceBuffer<F>>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_g2_tracegen(
            kind as u32,
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            source,
            d_operand_table.as_ptr().cast(),
            pc_base,
            pointer_max_bits,
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            d_bitwise_lookup.map_or(std::ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast()),
            d_range_tuple_checker.map_or(std::ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast()),
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen_reference(
        kind: DeltaAirKind,
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: Option<&DeviceBuffer<F>>,
        d_range_tuple_checker: Option<&DeviceBuffer<F>>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rvr_g2_tracegen_reference(
            kind as u32,
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr().cast(),
            pc_base,
            pointer_max_bits,
            d_range_checker.as_mut_ptr().cast(),
            d_range_checker.len() as u32,
            d_bitwise_lookup.map_or(std::ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast()),
            d_range_tuple_checker.map_or(std::ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast()),
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
    }
}

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
        fn _auipc_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_auipc_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _hintstore_decode_offsets(
            d_records: *const u8,
            records_len: usize,
            rows_used: usize,
            d_record_offsets: *mut OffsetInfo,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;

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

        fn _hintstore_replay_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_rows: *const u8,
            rows_used: usize,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            d_error: *mut u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn decode_offsets(
        d_records: &DeviceBuffer<u8>,
        records_len: usize,
        rows_used: usize,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        d_error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_hintstore_decode_offsets(
            d_records.as_ptr(),
            records_len,
            rows_used,
            d_record_offsets.as_mut_ptr(),
            d_error.as_mut_ptr(),
            stream,
        ))
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen_replay(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_rows: &DeviceBuffer<u8>,
        rows_used: usize,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        d_error: &DeviceBuffer<u32>,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_hintstore_replay_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_rows.as_ptr(),
            rows_used,
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            d_error.as_mut_ptr(),
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
        fn _jalr_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_jalr_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_less_than_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_less_than_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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

macro_rules! loadstore_compact_bitwise_tracegen {
    ($ffi:ident) => {
        extern "C" {
            fn $ffi(
                d_trace: *mut F,
                height: usize,
                width: usize,
                d_records: DeviceBufferView,
                d_operand_table: *const u8,
                pc_base: u32,
                pointer_max_bits: usize,
                d_range_checker: *mut u32,
                range_checker_num_bins: u32,
                d_bitwise_lookup: *mut u32,
                timestamp_max_bits: u32,
                stream: cudaStream_t,
            ) -> i32;
        }

        #[allow(clippy::too_many_arguments)]
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        pub unsafe fn tracegen_compact(
            d_trace: &DeviceBuffer<F>,
            height: usize,
            d_records: &DeviceBuffer<u8>,
            d_operand_table: &DeviceBuffer<u8>,
            pc_base: u32,
            pointer_max_bits: usize,
            d_range_checker: &DeviceBuffer<F>,
            d_bitwise_lookup: &DeviceBuffer<F>,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> Result<(), CudaError> {
            CudaError::from_result($ffi(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.view(),
                d_operand_table.as_ptr(),
                pc_base,
                pointer_max_bits,
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                timestamp_max_bits,
                stream,
            ))
        }
    };
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

    loadstore_compact_bitwise_tracegen!(_rv64_load_byte_tracegen_compact);

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

    loadstore_compact_bitwise_tracegen!(_rv64_store_byte_tracegen_compact);

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

    loadstore_compact_bitwise_tracegen!(_rv64_load_sign_extend_byte_tracegen_compact);

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
        fn _mul_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_mul_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_ptr() as *mut u32,
            d_range_tuple.as_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
            stream,
        ))
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
        pub fn _rv64_div_rem_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_div_rem_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_shift_logical_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_logical_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_shift_right_arithmetic_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_right_arithmetic_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _add_sub_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact alu3 wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_add_sub_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _addi_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_addi_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _bitwise_logic_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_bitwise_logic_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _jal_lui_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_jal_lui_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _beq_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_beq_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _blt_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): tracegen from compact wire records + the per-exe
    /// device operand table.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_blt_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _mulh_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        assert!(height.is_power_of_two() || height == 0);
        CudaError::from_result(_mulh_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_add_sub_w_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_add_sub_w_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_shift_w_logical_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
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
        fn _rv64_shift_w_right_arithmetic_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_logical_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_w_logical_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
    }

    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_right_arithmetic_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_shift_w_right_arithmetic_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
            stream,
        ))
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
        fn _rv64_mul_w_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        CudaError::from_result(_rv64_mul_w_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple.as_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
            stream,
        ))
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
        pub fn _rv64_div_rem_w_tracegen_compact(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_operand_table: *const u8,
            pc_base: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    /// M-GPUDEC (G2): compact-wire twin of `tracegen`.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "rvr")]
    pub unsafe fn tracegen_compact(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_operand_table: &DeviceBuffer<u8>,
        pc_base: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rv64_div_rem_w_tracegen_compact(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_operand_table.as_ptr(),
            pc_base,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
            stream,
        ))
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
