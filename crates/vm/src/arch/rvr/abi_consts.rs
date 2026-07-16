//! Compatibility guards for constants owned by external configurations.

use std::mem::{align_of, offset_of, size_of};

#[cfg(any(feature = "test-utils", feature = "cuda"))]
use openvm_instructions::VM_DIGEST_WIDTH;
use rvr_openvm_ext_ffi_common as ffi;

use crate::arch::rvr::preflight::{
    ChipRecordBuf, DeltaMemoryLogEntry, DeviceProgramEntry, MemoryLogEntry, PreflightTracerData,
    ProgramLogEntry, ProgramRunEntry, TouchedBlock, PREFLIGHT_INITIAL_TIMESTAMP,
    PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE,
    PREFLIGHT_TRACER_KIND,
};

#[cfg(any(feature = "test-utils", feature = "cuda"))]
const _: () =
    assert!(VM_DIGEST_WIDTH == openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE,);

// ── rvr-openvm-ext-ffi-common preflight-tracer layout ───────────────
const _: () = assert!(ffi::PREFLIGHT_TRACER_KIND == PREFLIGHT_TRACER_KIND);
const _: () = assert!(ffi::PREFLIGHT_INITIAL_TIMESTAMP == PREFLIGHT_INITIAL_TIMESTAMP);
const _: () = assert!(ffi::PREFLIGHT_MEMORY_KIND_READ == PREFLIGHT_MEMORY_KIND_READ);
const _: () = assert!(ffi::PREFLIGHT_MEMORY_KIND_WRITE == PREFLIGHT_MEMORY_KIND_WRITE);
const _: () = assert!(ffi::PREFLIGHT_MEMORY_KIND_TOUCH == PREFLIGHT_MEMORY_KIND_TOUCH);
const _: () = assert!(size_of::<ProgramLogEntry>() == ffi::PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE);
const _: () = assert!(align_of::<ProgramLogEntry>() == ffi::PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN);
const _: () = assert!(offset_of!(ProgramLogEntry, timestamp) == 0);
const _: () = assert!(offset_of!(ProgramLogEntry, pc_and_flags) == 4);
const _: () = assert!(offset_of!(ProgramLogEntry, write_value) == 8);
const _: () = assert!(size_of::<ProgramRunEntry>() == ffi::PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE);
const _: () = assert!(align_of::<ProgramRunEntry>() == ffi::PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN);
const _: () = assert!(offset_of!(ProgramRunEntry, first_pc) == 0);
const _: () = assert!(offset_of!(ProgramRunEntry, instruction_count) == 4);
const _: () = assert!(offset_of!(ProgramRunEntry, chronology_offset) == 8);
const _: () = assert!(offset_of!(ProgramRunEntry, complete) == 12);
const _: () = assert!(size_of::<DeviceProgramEntry>() == ffi::PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE);
const _: () =
    assert!(align_of::<DeviceProgramEntry>() == ffi::PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN);
const _: () = assert!(offset_of!(DeviceProgramEntry, pc) == 0);
const _: () = assert!(offset_of!(DeviceProgramEntry, filtered_index) == 4);
const _: () = assert!(size_of::<MemoryLogEntry>() == ffi::PREFLIGHT_MEMORY_LOG_ENTRY_SIZE);
const _: () = assert!(align_of::<MemoryLogEntry>() == ffi::PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN);
const _: () = assert!(offset_of!(MemoryLogEntry, timestamp) == 0);
const _: () = assert!(offset_of!(MemoryLogEntry, prev_timestamp) == 4);
const _: () = assert!(offset_of!(MemoryLogEntry, kind) == 8);
const _: () = assert!(offset_of!(MemoryLogEntry, addr_space) == 9);
const _: () = assert!(offset_of!(MemoryLogEntry, width) == 10);
const _: () = assert!(offset_of!(MemoryLogEntry, _pad0) == 11);
const _: () = assert!(offset_of!(MemoryLogEntry, address) == 16);
const _: () = assert!(offset_of!(MemoryLogEntry, value) == 24);
const _: () = assert!(offset_of!(MemoryLogEntry, prev_value) == 32);
const _: () =
    assert!(size_of::<DeltaMemoryLogEntry>() == ffi::PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE);
const _: () =
    assert!(align_of::<DeltaMemoryLogEntry>() == ffi::PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, timestamp) == 0);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, address) == 4);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, value) == 8);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, kind) == 16);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, addr_space) == 17);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, width) == 18);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, complete) == 19);
const _: () = assert!(offset_of!(DeltaMemoryLogEntry, _reserved) == 20);
const _: () = assert!(size_of::<TouchedBlock>() == ffi::PREFLIGHT_TOUCHED_BLOCK_SIZE);
const _: () = assert!(align_of::<TouchedBlock>() == ffi::PREFLIGHT_TOUCHED_BLOCK_ALIGN);
const _: () = assert!(offset_of!(TouchedBlock, addr_space) == 0);
const _: () = assert!(offset_of!(TouchedBlock, block_addr) == 4);
const _: () = assert!(offset_of!(TouchedBlock, initial_value) == 8);
const _: () = assert!(size_of::<PreflightTracerData>() == ffi::PREFLIGHT_TRACER_DATA_SIZE);
const _: () = assert!(align_of::<PreflightTracerData>() == ffi::PREFLIGHT_TRACER_DATA_ALIGN);
const _: () = assert!(offset_of!(PreflightTracerData, program_log) == 0);
const _: () = assert!(offset_of!(PreflightTracerData, memory_log) == 8);
const _: () = assert!(offset_of!(PreflightTracerData, chip_counts) == 16);
const _: () = assert!(offset_of!(PreflightTracerData, shadow_register) == 24);
const _: () = assert!(offset_of!(PreflightTracerData, shadow_memory) == 32);
const _: () = assert!(offset_of!(PreflightTracerData, shadow_public_values) == 40);
const _: () = assert!(offset_of!(PreflightTracerData, public_values_base) == 48);
const _: () = assert!(offset_of!(PreflightTracerData, touched) == 56);
const _: () = assert!(offset_of!(PreflightTracerData, program_log_len) == 64);
const _: () = assert!(offset_of!(PreflightTracerData, memory_log_len) == 68);
const _: () = assert!(offset_of!(PreflightTracerData, program_log_cap) == 72);
const _: () = assert!(offset_of!(PreflightTracerData, memory_log_cap) == 76);
const _: () = assert!(offset_of!(PreflightTracerData, chip_counts_len) == 80);
const _: () = assert!(offset_of!(PreflightTracerData, touched_len) == 84);
const _: () = assert!(offset_of!(PreflightTracerData, touched_cap) == 88);
const _: () = assert!(offset_of!(PreflightTracerData, timestamp) == 92);
const _: () = assert!(offset_of!(PreflightTracerData, chip_records) == 96);
const _: () = assert!(offset_of!(PreflightTracerData, exec_frequencies) == 104);
const _: () = assert!(offset_of!(PreflightTracerData, exec_frequencies_len) == 112);
const _: () = assert!(offset_of!(PreflightTracerData, delta_records) == 120);
const _: () = assert!(offset_of!(PreflightTracerData, custom_memory_scratch) == 128);
const _: () = assert!(offset_of!(PreflightTracerData, custom_memory_scratch_len) == 136);
const _: () = assert!(offset_of!(PreflightTracerData, custom_memory_scratch_cap) == 140);
const _: () = assert!(offset_of!(PreflightTracerData, chip_counts_touched) == 144);
const _: () = assert!(offset_of!(PreflightTracerData, chip_counts_touched_len) == 152);
const _: () = assert!(offset_of!(PreflightTracerData, chip_counts_touched_cap) == 156);
const _: () = assert!(offset_of!(PreflightTracerData, exec_frequencies_touched) == 160);
const _: () = assert!(offset_of!(PreflightTracerData, exec_frequencies_touched_len) == 168);
const _: () = assert!(offset_of!(PreflightTracerData, exec_frequencies_touched_cap) == 172);
const _: () = assert!(offset_of!(PreflightTracerData, native_detail) == 176);
const _: () = assert!(offset_of!(PreflightTracerData, device_aux_patches) == 184);
const _: () = assert!(offset_of!(PreflightTracerData, device_aux_references) == 192);
const _: () = assert!(offset_of!(PreflightTracerData, dirty_memory_pages) == 200);
const _: () = assert!(offset_of!(PreflightTracerData, device_aux_patches_len) == 208);
const _: () = assert!(offset_of!(PreflightTracerData, device_aux_patches_cap) == 212);
const _: () = assert!(offset_of!(PreflightTracerData, device_aux_references_cap) == 216);
const _: () = assert!(offset_of!(PreflightTracerData, dirty_memory_pages_words) == 220);
const _: () = assert!(offset_of!(PreflightTracerData, program_runs) == 224);
const _: () = assert!(offset_of!(PreflightTracerData, device_program_references) == 232);
const _: () = assert!(offset_of!(PreflightTracerData, program_runs_len) == 240);
const _: () = assert!(offset_of!(PreflightTracerData, program_runs_cap) == 244);
const _: () = assert!(offset_of!(PreflightTracerData, program_instruction_len) == 248);
const _: () = assert!(offset_of!(PreflightTracerData, device_program_references_len) == 252);
const _: () = assert!(offset_of!(PreflightTracerData, device_program_references_cap) == 256);
const _: () = assert!(offset_of!(PreflightTracerData, g2) == 264);
const _: () = assert!(size_of::<ChipRecordBuf>() == ffi::PREFLIGHT_CHIP_RECORD_BUF_SIZE);
const _: () = assert!(align_of::<ChipRecordBuf>() == ffi::PREFLIGHT_CHIP_RECORD_BUF_ALIGN);
const _: () = assert!(offset_of!(ChipRecordBuf, base) == 0);
const _: () = assert!(offset_of!(ChipRecordBuf, len) == 8);
const _: () = assert!(offset_of!(ChipRecordBuf, cap) == 12);
const _: () = assert!(offset_of!(ChipRecordBuf, stride) == 16);
const _: () = assert!(offset_of!(ChipRecordBuf, core_off) == 20);
const _: () = assert!(offset_of!(ChipRecordBuf, flags) == 24);
