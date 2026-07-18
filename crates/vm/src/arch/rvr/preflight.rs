//! Preflight tracer ABI mirror for rvr-generated native execution.

use std::{collections::BTreeMap, mem::MaybeUninit, sync::Arc};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::{
    exe::VmExe,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    PUBLIC_VALUES_AS,
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rvr_openvm_lift::{RvrRuntimeExtension, TraceChipIndex};

use super::{
    bridge::{map_rvr_execute_error, public_values_slice},
    compile::{ChipMapping, RvrCompiled},
    execute::execute_preflight as execute_preflight_raw,
    g2::{next_segment_id, RvrG2CapacitiesV1, RvrG2PreparedV1, RvrG2SegmentV1},
    preflight_normalizer::{
        build_preflight_replay_with_scratch, PreflightMemoryAccessAux, PreflightMemoryReplay,
        PreflightShadowsView, WORD_BYTES,
    },
    preflight_pool::{scrub_shadows, RvrPreflightBufferPool},
};
use crate::{
    arch::{
        interpreter_preflight::PreflightInterpretedInstance, ExecutionError, ExecutionState,
        Streams, SystemConfig, VmState, BLOCK_FE_WIDTH,
    },
    system::{
        memory::online::{GuestMemory, LinearMemory, PAGE_SIZE},
        SystemRecords,
    },
};

pub const PREFLIGHT_TRACER_KIND: u32 = rvr_openvm_ext_ffi_common::PREFLIGHT_TRACER_KIND;
pub const PREFLIGHT_INITIAL_TIMESTAMP: u32 = rvr_openvm_ext_ffi_common::PREFLIGHT_INITIAL_TIMESTAMP;
const PREFLIGHT_CUSTOM_MEMORY_SCRATCH_CAP: usize = 256;
pub const PREFLIGHT_MEMORY_KIND_READ: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_READ;
pub const PREFLIGHT_MEMORY_KIND_WRITE: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_WRITE;
pub const PREFLIGHT_MEMORY_KIND_TOUCH: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_TOUCH;
pub const PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL;
pub const PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW;
pub const PREFLIGHT_CHIP_RECORD_FLAG_RESIDUAL_MEMORY_CHRONOLOGY: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_RESIDUAL_MEMORY_CHRONOLOGY;
pub const PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS;
pub const PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE;
pub const PREFLIGHT_CHIP_RECORD_FLAG_COMPACT_RESIDUAL_MEMORY: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_COMPACT_RESIDUAL_MEMORY;
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX;
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX_ORACLE: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX_ORACLE;
pub const PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_CHRONOLOGY: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_CHRONOLOGY;
/// R3: byte size of one compact AddSub inline record (see the C
/// `PreflightAddSubRecord`). Used to size the per-chip inline-record buffers.
pub const PREFLIGHT_ADDSUB_RECORD_SIZE: usize =
    rvr_openvm_ext_ffi_common::PREFLIGHT_ADDSUB_RECORD_SIZE;
/// R3 compact record strides for the branch / write-only / read+write shapes.
pub const PREFLIGHT_BRANCH2_RECORD_SIZE: usize =
    rvr_openvm_ext_ffi_common::PREFLIGHT_BRANCH2_RECORD_SIZE;
pub const PREFLIGHT_WR1_RECORD_SIZE: usize = rvr_openvm_ext_ffi_common::PREFLIGHT_WR1_RECORD_SIZE;
pub const PREFLIGHT_RW1_RECORD_SIZE: usize = rvr_openvm_ext_ffi_common::PREFLIGHT_RW1_RECORD_SIZE;
pub const PREFLIGHT_DELTA_RECORD_SIZE: usize =
    rvr_openvm_ext_ffi_common::PREFLIGHT_DELTA_RECORD_SIZE;

const RVR_NATIVE_DETAIL_PHASES: usize = 7;
const RVR_NATIVE_DETAIL_FAMILIES: usize = 9;

/// Profiling-only counters shared with the generated preflight C. The pointer
/// in [`PreflightTracerData`] is null on every normal execution route.
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct RvrNativeDetail {
    pub family_cycles: [u64; RVR_NATIVE_DETAIL_FAMILIES],
    pub family_instructions: [u64; RVR_NATIVE_DETAIL_FAMILIES],
    pub phase_cycles: [u64; RVR_NATIVE_DETAIL_PHASES],
    pub phase_samples: [u64; RVR_NATIVE_DETAIL_PHASES],
    pub phase_events: [u64; RVR_NATIVE_DETAIL_PHASES],
    pub phase_bytes: [u64; RVR_NATIVE_DETAIL_PHASES],
    pub family_started: u64,
    pub outer_started: u64,
    pub timer_overhead: u64,
    pub sample_state: u32,
    pub sample_countdown: u32,
    pub current_family: u32,
    pub family_active: u32,
}

impl RvrNativeDetail {
    fn new() -> Self {
        Self {
            timer_overhead: calibrate_native_detail_clock(),
            sample_state: 0x9e37_79b9,
            sample_countdown: 769,
            current_family: u32::MAX,
            ..Self::default()
        }
    }

    fn start(&mut self) {
        let started = native_detail_clock();
        self.sample_state ^= started as u32;
        self.sample_countdown = 512 + (self.sample_state & 1023);
        self.outer_started = started;
    }

    fn finish(&mut self) -> u64 {
        let finished = native_detail_clock();
        if self.family_active != 0 {
            let family = self.current_family as usize;
            if family < self.family_cycles.len() {
                self.family_cycles[family] = self.family_cycles[family]
                    .saturating_add(finished.saturating_sub(self.family_started));
            }
            self.family_active = 0;
        }
        finished.saturating_sub(self.outer_started)
    }
}

#[inline]
fn native_detail_clock() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(target_arch = "aarch64")]
    {
        let value: u64;
        unsafe {
            core::arch::asm!("isb", "mrs {value}, cntvct_el0", value = out(reg) value);
        }
        value
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

fn calibrate_native_detail_clock() -> u64 {
    (0..10_000)
        .map(|_| {
            let started = native_detail_clock();
            native_detail_clock().saturating_sub(started)
        })
        .min()
        .unwrap_or(0)
}

/// C-compatible preflight program log entry.
///
/// Layout matches `ProgramLogEntry` in `openvm_tracer_preflight.h`.
/// OpenVM pcs are four-byte aligned, so the low two bits of `pc_and_flags`
/// carry fail-closed side-band guards without enlarging this side log.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramLogEntry {
    pub timestamp: u32,
    pub pc_and_flags: u32,
    /// Post-write register block for arena-native W-family chronology.
    pub write_value: u64,
}

impl ProgramLogEntry {
    const WRITE_COMPLETE: u32 = 1;
    const CROSSING_RESIDUAL: u32 = 2;
    const FLAGS: u32 = Self::WRITE_COMPLETE | Self::CROSSING_RESIDUAL;

    #[inline]
    pub fn new(timestamp: u32, pc: u32) -> Self {
        debug_assert_eq!(pc & 3, 0, "OpenVM pc must be four-byte aligned");
        Self {
            timestamp,
            pc_and_flags: pc,
            write_value: 0,
        }
    }

    #[inline]
    pub fn pc(&self) -> u32 {
        self.pc_and_flags & !Self::FLAGS
    }

    #[inline]
    pub fn write_complete(&self) -> bool {
        self.pc_and_flags & Self::WRITE_COMPLETE != 0
    }

    /// The inline compact witness is only a chronology placeholder for this
    /// row; both memory blocks are carried by residual memory events.
    #[inline]
    pub fn crossing_residual(&self) -> bool {
        self.pc_and_flags & Self::CROSSING_RESIDUAL != 0
    }
}

/// One all-direct device-chronology descriptor emitted at basic-block entry.
/// The device expands `instruction_count` consecutive program counters into
/// `chronology_offset..` and derives their filtered program indices from the
/// immutable operand table. `complete == 1` is a fail-closed ABI guard.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramRunEntry {
    pub first_pc: u32,
    pub instruction_count: u32,
    pub chronology_offset: u32,
    pub complete: u32,
}

/// Oracle-only host expansion of one executed instruction. Production writes
/// no entries; CUDA reconstructs this complete vector from [`ProgramRunEntry`]
/// and uses it to build the execution-frequency table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceProgramEntry {
    pub pc: u32,
    pub filtered_index: u32,
}

/// C-compatible preflight memory log entry.
///
/// Layout matches `MemoryLogEntry` in `openvm_tracer_preflight.h`.
///
/// R1: self-contained. `prev_timestamp` is the block's previous-access
/// timestamp (from the C timestamp shadow) and `prev_value` is the block's
/// value before this access (only meaningful for writes). Together they let the
/// host build memory-record aux data in a single linear pass without replaying
/// the log.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryLogEntry {
    pub timestamp: u32,
    pub prev_timestamp: u32,
    pub kind: u8,
    pub addr_space: u8,
    pub width: u8,
    pub _pad0: u8,
    pub _pad1: u32,
    pub address: u64,
    pub value: u64,
    pub prev_value: u64,
}

/// Delta-only residual-memory wire. Previous timestamp/value are reconstructed
/// by the chronological decoder; address is narrowed to OpenVM's u32 pointer
/// domain. `complete == 1` and `_reserved == 0` are fail-closed ABI guards.
/// CPU, compact, partial-direct, and non-delta routes retain `MemoryLogEntry`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeltaMemoryLogEntry {
    pub timestamp: u32,
    pub address: u32,
    pub value: u64,
    pub kind: u8,
    pub addr_space: u8,
    pub width: u8,
    pub complete: u8,
    pub _reserved: u32,
}

/// C-compatible preflight touched-block entry.
///
/// Layout matches `TouchedBlock` in `openvm_tracer_preflight.h`. Records a block
/// touched for the first time this segment. Host routes finalize
/// `touched_memory` from the live memory + shadow. The production all-direct
/// CUDA delta route leaves this empty and seeds replay from device-resident
/// segment-start memory; its equality oracle retains these host references.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TouchedBlock {
    pub addr_space: u32,
    pub block_addr: u32,
    /// Block contents immediately before this segment's first access.
    pub initial_value: u64,
}

/// Legacy-host predecessor values retained only by the device replay oracle.
/// There is one entry per compact residual-memory event.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAuxReference {
    pub prev_timestamp: u32,
    pub _reserved: u32,
    pub prev_value: u64,
}

/// One custom direct-final field that must consume a device-reconstructed
/// predecessor. `target` points into a staged dense-arena backing whose
/// allocation remains stable until system trace generation runs the replay.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAuxPatch {
    pub target: u64,
    pub event_index: u32,
    pub kind: u32,
    /// Legacy-host value in oracle mode, zero in production.
    pub expected: u64,
}

/// Complete C-written custom arena retained only by the device replay oracle.
/// `base` stays valid until the staged arena is installed into the returned
/// record-arena vector and consumed after system-first device replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceAuxArenaReference {
    pub air_idx: usize,
    pub base: u64,
    pub expected: Vec<u8>,
}

pub const DEVICE_AUX_PATCH_U32: u32 = 0;
pub const DEVICE_AUX_PATCH_U64: u32 = 1;
const DEVICE_AUX_EVENT_INDEX_LIMIT: usize = 1usize << 31;

/// C-compatible per-chip inline-record buffer descriptor (R3/R4).
///
/// Layout must match the C `ChipRecordBuf` in `openvm_tracer_preflight.h`.
/// `base` points at a host-provided byte buffer for one chip; `len` is the
/// byte cursor the C advances by `stride` per record; `cap` the byte capacity
/// (a multiple of `stride`), so record i sits at `base + i*stride`.
/// Compact-wire buffers set `stride` to the packed record size; arena-native
/// buffers set it to the arena row/record pitch, with `base` 32-aligned by
/// host contract so a zero-copy [`DenseRecordArena`](crate::arch::DenseRecordArena)
/// adopt cannot slice a shifted range. A null `base` means the chip is not
/// migrated to inline records (it uses the verbose memory log).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChipRecordBuf {
    pub base: *mut u8,
    pub len: u32,
    pub cap: u32,
    pub stride: u32,
    /// R4 arena-native: byte offset of the core record within each record
    /// slot (adapter fields sit at offset 0); the per-flavor arena geometry
    /// (Matrix: adapter trace width; Dense: aligned adapter record size) is
    /// computed host-side, so one generated .so serves both arena flavors.
    /// Zero in compact-wire mode.
    pub core_off: u32,
    /// ZG2 transport flags. `DIRECT_FINAL` means the C writer targets the
    /// consumer backing and no host adoption/assembly may follow.
    pub flags: u32,
}

impl Default for ChipRecordBuf {
    fn default() -> Self {
        Self {
            base: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            stride: 0,
            core_off: 0,
            flags: 0,
        }
    }
}

const _: () = assert!(
    std::mem::size_of::<ChipRecordBuf>()
        == rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_BUF_SIZE
);

/// C-compatible preflight tracer data.
///
/// Layout must exactly match the C `Tracer` struct in
/// `openvm_tracer_preflight.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PreflightTracerData {
    pub program_log: *mut ProgramLogEntry,
    pub memory_log: *mut MemoryLogEntry,
    pub chip_counts: *mut u32,
    /// Per-address-space last-access timestamp shadows, indexed by block index
    /// (`block_addr / WORD_SIZE`). 0 means untouched this segment.
    pub shadow_register: *mut u32,
    pub shadow_memory: *mut u32,
    pub shadow_public_values: *mut u32,
    /// Public-values byte buffer, aliased so reveal writes can read the block's
    /// previous value.
    pub public_values_base: *mut u8,
    pub touched: *mut TouchedBlock,
    pub program_log_len: u32,
    pub memory_log_len: u32,
    pub program_log_cap: u32,
    pub memory_log_cap: u32,
    pub chip_counts_len: u32,
    pub touched_len: u32,
    pub touched_cap: u32,
    pub timestamp: u32,
    /// Array of `chip_counts_len` per-chip inline-record buffers (R3), indexed
    /// by chip (AIR) index. Null until [`PreflightTracerData::set_chip_records`]
    /// attaches it; a null array (or a null per-chip `base`) means no chip uses
    /// inline records and every opcode takes the verbose-log path.
    pub chip_records: *mut ChipRecordBuf,
    /// Direct execution-frequency counters indexed by the compile-time
    /// filtered program index.
    pub exec_frequencies: *mut u32,
    pub exec_frequencies_len: u32,
    /// Stage-2 global chronological delta-record target. This is separate
    /// from the per-AIR compact targets because its cross-AIR order is what
    /// makes previous timestamps implicit.
    pub delta_records: *mut ChipRecordBuf,
    /// Bounded instruction-local predecessor capture for custom direct-final
    /// emitters while the persistent residual-memory log uses its compact
    /// 24-byte schema.
    pub custom_memory_scratch: *mut MemoryLogEntry,
    /// `u32::MAX` means inactive; otherwise this is the written scratch prefix.
    pub custom_memory_scratch_len: u32,
    pub custom_memory_scratch_cap: u32,
    /// Write-only lists of the indices whose counter changed 0→nonzero in
    /// this segment. The consumer scrubs exactly these indices before pooled
    /// reuse, avoiding a full counter-table memset or scan.
    pub chip_counts_touched: *mut u32,
    pub chip_counts_touched_len: u32,
    pub chip_counts_touched_cap: u32,
    pub exec_frequencies_touched: *mut u32,
    pub exec_frequencies_touched_len: u32,
    pub exec_frequencies_touched_cap: u32,
    /// Profiling-only detailed counters. Null unless explicitly enabled.
    pub native_detail: *mut RvrNativeDetail,
    /// Custom direct-final predecessor fields to patch after device replay.
    pub device_aux_patches: *mut DeviceAuxPatch,
    /// Legacy-host predecessor vector, present only in fail-hard oracle mode.
    pub device_aux_references: *mut DeviceAuxReference,
    /// Per-segment main-memory dirty-page bitmap. This preserves sparse H2D
    /// completeness without retaining the host first-touch shadow.
    pub dirty_memory_pages: *mut u64,
    pub device_aux_patches_len: u32,
    pub device_aux_patches_cap: u32,
    pub device_aux_references_cap: u32,
    pub dirty_memory_pages_words: u32,
    /// L2 block-run chronology, present only on the all-direct device route.
    pub program_runs: *mut ProgramRunEntry,
    /// Legacy per-instruction chronology retained only by the equality oracle.
    pub device_program_references: *mut DeviceProgramEntry,
    pub program_runs_len: u32,
    pub program_runs_cap: u32,
    pub program_instruction_len: u32,
    pub device_program_references_len: u32,
    pub device_program_references_cap: u32,
    /// G2 private compact-wire producer. Null on every established route.
    pub g2: *mut rvr_openvm_ext_ffi_common::G2ProducerV1,
}

impl PreflightTracerData {
    /// Build a tracer over the log/chip buffers with the shadow pointers left
    /// null; call [`PreflightTracerData::set_shadows`] before executing.
    pub fn new(
        program_log: &mut [ProgramLogEntry],
        memory_log: &mut [MemoryLogEntry],
        chip_counts: &mut [u32],
    ) -> Self {
        Self {
            program_log: program_log.as_mut_ptr(),
            memory_log: memory_log.as_mut_ptr(),
            chip_counts: chip_counts.as_mut_ptr(),
            shadow_register: std::ptr::null_mut(),
            shadow_memory: std::ptr::null_mut(),
            shadow_public_values: std::ptr::null_mut(),
            public_values_base: std::ptr::null_mut(),
            touched: std::ptr::null_mut(),
            program_log_len: 0,
            memory_log_len: 0,
            program_log_cap: program_log.len() as u32,
            memory_log_cap: memory_log.len() as u32,
            chip_counts_len: chip_counts.len() as u32,
            touched_len: 0,
            touched_cap: 0,
            timestamp: PREFLIGHT_INITIAL_TIMESTAMP,
            chip_records: std::ptr::null_mut(),
            exec_frequencies: std::ptr::null_mut(),
            exec_frequencies_len: 0,
            delta_records: std::ptr::null_mut(),
            custom_memory_scratch: std::ptr::null_mut(),
            custom_memory_scratch_len: u32::MAX,
            custom_memory_scratch_cap: 0,
            chip_counts_touched: std::ptr::null_mut(),
            chip_counts_touched_len: 0,
            chip_counts_touched_cap: 0,
            exec_frequencies_touched: std::ptr::null_mut(),
            exec_frequencies_touched_len: 0,
            exec_frequencies_touched_cap: 0,
            native_detail: std::ptr::null_mut(),
            device_aux_patches: std::ptr::null_mut(),
            device_aux_references: std::ptr::null_mut(),
            dirty_memory_pages: std::ptr::null_mut(),
            device_aux_patches_len: 0,
            device_aux_patches_cap: 0,
            device_aux_references_cap: 0,
            dirty_memory_pages_words: 0,
            program_runs: std::ptr::null_mut(),
            device_program_references: std::ptr::null_mut(),
            program_runs_len: 0,
            program_runs_cap: 0,
            program_instruction_len: 0,
            device_program_references_len: 0,
            device_program_references_cap: 0,
            g2: std::ptr::null_mut(),
        }
    }

    /// Build a tracer over uninitialized log buffers. The C tracer only ever
    /// writes these buffers and the host only reads the written prefix (the
    /// `*_len` counters), so the backing stores never need the zero-fill —
    /// which is a real per-call cost at the multi-GB capacities of large
    /// segments. Shadow pointers are left null; call
    /// [`PreflightTracerData::set_shadows`] before executing.
    pub fn new_uninit(
        program_log: &mut [MaybeUninit<ProgramLogEntry>],
        memory_log: &mut [MaybeUninit<MemoryLogEntry>],
        chip_counts: &mut [u32],
    ) -> Self {
        Self {
            program_log: program_log.as_mut_ptr().cast(),
            memory_log: memory_log.as_mut_ptr().cast(),
            chip_counts: chip_counts.as_mut_ptr(),
            program_log_cap: program_log.len() as u32,
            memory_log_cap: memory_log.len() as u32,
            chip_counts_len: chip_counts.len() as u32,
            ..Self::default()
        }
    }

    /// Attach an uninitialized touched-block buffer (see
    /// [`PreflightTracerData::new_uninit`]; write-only from C, prefix-read by
    /// the host). Must be called after [`PreflightTracerData::set_shadows`],
    /// which also sets the (initialized) touched buffer fields.
    pub fn set_touched_uninit(&mut self, touched: &mut [MaybeUninit<TouchedBlock>]) {
        self.touched = touched.as_mut_ptr().cast();
        self.touched_len = 0;
        self.touched_cap = touched.len() as u32;
    }

    /// Attach the per-chip inline-record buffers (R3). The slice must have one
    /// entry per chip (`chip_counts_len`); migrated chips carry a non-null
    /// `base` pointing at a metered-height-sized record buffer, others null.
    pub fn set_chip_records(&mut self, chip_records: &mut [ChipRecordBuf]) {
        debug_assert_eq!(chip_records.len() as u32, self.chip_counts_len);
        self.chip_records = chip_records.as_mut_ptr();
    }

    pub fn set_exec_frequencies(&mut self, frequencies: &mut [u32]) {
        self.exec_frequencies = frequencies.as_mut_ptr();
        self.exec_frequencies_len = frequencies.len() as u32;
    }

    /// Attach write-only first-touch index buffers for both counter arrays.
    /// Each capacity equals its counter table length, so one entry per unique
    /// 0→nonzero transition cannot overflow under the counter helper contract.
    pub fn set_counter_touched_uninit(
        &mut self,
        chip_counts_touched: &mut [MaybeUninit<u32>],
        exec_frequencies_touched: &mut [MaybeUninit<u32>],
    ) {
        debug_assert_eq!(chip_counts_touched.len() as u32, self.chip_counts_len);
        debug_assert_eq!(
            exec_frequencies_touched.len() as u32,
            self.exec_frequencies_len
        );
        self.chip_counts_touched = chip_counts_touched.as_mut_ptr().cast();
        self.chip_counts_touched_len = 0;
        self.chip_counts_touched_cap = chip_counts_touched.len() as u32;
        self.exec_frequencies_touched = exec_frequencies_touched.as_mut_ptr().cast();
        self.exec_frequencies_touched_len = 0;
        self.exec_frequencies_touched_cap = exec_frequencies_touched.len() as u32;
    }

    pub fn set_delta_records(&mut self, delta_records: &mut ChipRecordBuf) {
        self.delta_records = delta_records;
    }

    fn set_custom_memory_scratch(&mut self, scratch: &mut [MaybeUninit<MemoryLogEntry>]) {
        self.custom_memory_scratch = scratch.as_mut_ptr().cast();
        self.custom_memory_scratch_len = u32::MAX;
        self.custom_memory_scratch_cap = scratch.len() as u32;
    }

    fn set_native_detail(&mut self, detail: &mut RvrNativeDetail) {
        self.native_detail = detail;
    }

    fn set_device_aux(
        &mut self,
        patches: &mut [MaybeUninit<DeviceAuxPatch>],
        references: &mut [MaybeUninit<DeviceAuxReference>],
        dirty_memory_pages: &mut [u64],
    ) {
        self.device_aux_patches = patches.as_mut_ptr().cast();
        self.device_aux_references = references.as_mut_ptr().cast();
        self.dirty_memory_pages = dirty_memory_pages.as_mut_ptr();
        self.device_aux_patches_len = 0;
        self.device_aux_patches_cap = patches.len() as u32;
        self.device_aux_references_cap = references.len() as u32;
        self.dirty_memory_pages_words = dirty_memory_pages.len() as u32;
    }

    fn set_device_chronology(
        &mut self,
        runs: &mut [MaybeUninit<ProgramRunEntry>],
        references: &mut [MaybeUninit<DeviceProgramEntry>],
    ) {
        self.program_runs = runs.as_mut_ptr().cast();
        self.device_program_references = references.as_mut_ptr().cast();
        self.program_runs_len = 0;
        self.program_runs_cap = runs.len() as u32;
        self.program_instruction_len = 0;
        self.device_program_references_len = 0;
        self.device_program_references_cap = references.len() as u32;
    }

    fn set_g2(&mut self, producer: &mut rvr_openvm_ext_ffi_common::G2ProducerV1) {
        self.g2 = producer;
    }

    /// Reuse the tracer's memory pointer/cursor for the delta-only compact
    /// residual schema. Generated C selects this layout only when the delta
    /// target also carries `COMPACT_RESIDUAL_MEMORY`.
    pub fn set_delta_memory_log(&mut self, memory_log: &mut [MaybeUninit<DeltaMemoryLogEntry>]) {
        self.memory_log = memory_log.as_mut_ptr().cast();
        self.memory_log_len = 0;
        self.memory_log_cap = memory_log.len() as u32;
    }

    /// Attach the per-address-space timestamp shadows, the public-values base
    /// pointer, and the touched-block buffer. The shadow slices must be sized to
    /// each address space's block count and zero-initialized.
    #[allow(clippy::too_many_arguments)]
    pub fn set_shadows(
        &mut self,
        shadow_register: &mut [u32],
        shadow_memory: &mut [u32],
        shadow_public_values: &mut [u32],
        public_values_base: *mut u8,
        touched: &mut [TouchedBlock],
    ) {
        self.shadow_register = shadow_register.as_mut_ptr();
        self.shadow_memory = shadow_memory.as_mut_ptr();
        self.shadow_public_values = shadow_public_values.as_mut_ptr();
        self.public_values_base = public_values_base;
        self.touched = touched.as_mut_ptr();
        self.touched_len = 0;
        self.touched_cap = touched.len() as u32;
    }
}

impl Default for PreflightTracerData {
    fn default() -> Self {
        Self {
            program_log: std::ptr::null_mut(),
            memory_log: std::ptr::null_mut(),
            chip_counts: std::ptr::null_mut(),
            shadow_register: std::ptr::null_mut(),
            shadow_memory: std::ptr::null_mut(),
            shadow_public_values: std::ptr::null_mut(),
            public_values_base: std::ptr::null_mut(),
            touched: std::ptr::null_mut(),
            program_log_len: 0,
            memory_log_len: 0,
            program_log_cap: 0,
            memory_log_cap: 0,
            chip_counts_len: 0,
            touched_len: 0,
            touched_cap: 0,
            timestamp: PREFLIGHT_INITIAL_TIMESTAMP,
            chip_records: std::ptr::null_mut(),
            exec_frequencies: std::ptr::null_mut(),
            exec_frequencies_len: 0,
            delta_records: std::ptr::null_mut(),
            custom_memory_scratch: std::ptr::null_mut(),
            custom_memory_scratch_len: u32::MAX,
            custom_memory_scratch_cap: 0,
            chip_counts_touched: std::ptr::null_mut(),
            chip_counts_touched_len: 0,
            chip_counts_touched_cap: 0,
            exec_frequencies_touched: std::ptr::null_mut(),
            exec_frequencies_touched_len: 0,
            exec_frequencies_touched_cap: 0,
            native_detail: std::ptr::null_mut(),
            device_aux_patches: std::ptr::null_mut(),
            device_aux_references: std::ptr::null_mut(),
            dirty_memory_pages: std::ptr::null_mut(),
            device_aux_patches_len: 0,
            device_aux_patches_cap: 0,
            device_aux_references_cap: 0,
            dirty_memory_pages_words: 0,
            program_runs: std::ptr::null_mut(),
            device_program_references: std::ptr::null_mut(),
            program_runs_len: 0,
            program_runs_cap: 0,
            program_instruction_len: 0,
            device_program_references_len: 0,
            device_program_references_cap: 0,
            g2: std::ptr::null_mut(),
        }
    }
}

pub type PreflightTracer = *mut PreflightTracerData;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightRawLogs {
    pub program_log: Vec<ProgramLogEntry>,
    /// All-direct device chronology: one descriptor per executed basic block.
    pub program_runs: Vec<ProgramRunEntry>,
    /// Oracle-only full per-instruction host chronology.
    pub device_program_references: Vec<DeviceProgramEntry>,
    pub memory_log: Vec<MemoryLogEntry>,
    /// Populated only by the all-direct CUDA delta route. Mutually exclusive
    /// with `memory_log`; all other callers keep the full host schema.
    pub delta_memory_log: Vec<DeltaMemoryLogEntry>,
    pub chip_counts: Vec<u32>,
    /// Counter slots first touched this segment; used only to scrub the
    /// pooled `chip_counts` allocation after its consumer is finished.
    pub chip_counts_touched: Vec<u32>,
    /// First-touch seeds retained until delta replay has completed.
    pub touched: Vec<TouchedBlock>,
    /// Custom direct-final predecessor fields that must be populated from the
    /// device replay before extension trace generation consumes their arenas.
    pub device_aux_patches: Vec<DeviceAuxPatch>,
    /// Legacy-host predecessor vector, populated only by the equality oracle.
    pub device_aux_references: Vec<DeviceAuxReference>,
    /// Full custom arenas before their predecessor fields are device-patched.
    pub device_aux_arena_references: Vec<DeviceAuxArenaReference>,
}

/// One migrated chip's inline compact records for a segment, written by the
/// generated C (R3). `bytes` holds `record_size`-strided compact records in
/// program-log order; host record assembly expands each into the chip's full
/// record, re-deriving program-redundant operands from the instruction at the
/// record's `from_pc`.
#[derive(Debug)]
pub struct RvrInlineChipRecords {
    pub air_idx: usize,
    /// Compact record stride in bytes (from the compile metadata).
    pub record_size: usize,
    pub bytes: Vec<u8>,
}

/// Owned Stage-2 chronological record backing. `offset` preserves the
/// original allocation while exposing a 32-byte-aligned C/decoder window;
/// CUDA returns `backing` to the page-locked pool after device decode.
pub struct RvrDeltaRecords {
    backing: Option<Vec<u8>>,
    pub(crate) offset: usize,
    pub(crate) written: usize,
    dirty_len: usize,
    needs_prefault: bool,
    pool: RvrPreflightBufferPool,
}

impl RvrDeltaRecords {
    fn new(
        backing: Vec<u8>,
        offset: usize,
        capacity: usize,
        pool: RvrPreflightBufferPool,
        needs_prefault: bool,
    ) -> Self {
        Self {
            backing: Some(backing),
            offset,
            written: 0,
            // Until native execution returns, conservatively assume C may
            // have dirtied the complete target on an error path.
            dirty_len: offset + capacity,
            needs_prefault,
            pool,
        }
    }

    fn aligned_mut_ptr(&mut self) -> *mut u8 {
        unsafe {
            self.backing
                .as_mut()
                .expect("delta backing already returned")
                .as_mut_ptr()
                .add(self.offset)
        }
    }

    fn set_written(&mut self, written: usize) {
        self.written = written;
        self.dirty_len = self.offset + written;
    }

    pub fn bytes(&self) -> &[u8] {
        &self
            .backing
            .as_ref()
            .expect("delta backing already returned")[self.offset..self.offset + self.written]
    }

    /// Return the host chronology inputs after a device decoder has
    /// synchronized their H2D copies. They originate from the same compiled
    /// executor pool as this delta backing and would otherwise be freed when
    /// the device-bound segment is dropped.
    #[doc(hidden)]
    pub fn recycle_device_inputs(
        &self,
        program_log: Vec<ProgramLogEntry>,
        program_runs: Vec<ProgramRunEntry>,
        device_program_references: Vec<DeviceProgramEntry>,
        memory_log: Vec<MemoryLogEntry>,
        delta_memory_log: Vec<DeltaMemoryLogEntry>,
        touched: Vec<TouchedBlock>,
    ) {
        self.pool.recycle_segment_buffers(
            PreflightRawLogs {
                program_log,
                program_runs,
                device_program_references,
                memory_log,
                delta_memory_log,
                chip_counts: Vec::new(),
                chip_counts_touched: Vec::new(),
                touched,
                device_aux_patches: Vec::new(),
                device_aux_references: Vec::new(),
                device_aux_arena_references: Vec::new(),
            },
            Vec::new(),
            None,
        );
    }
}

impl Drop for RvrDeltaRecords {
    fn drop(&mut self) {
        if let Some(backing) = self.backing.take() {
            self.pool.recycle_delta_backing(backing, self.dirty_len);
        }
    }
}

pub struct RvrPreflightOutput<F> {
    pub system_records: SystemRecords<F>,
    pub raw_logs: PreflightRawLogs,
    pub access_aux: Vec<PreflightMemoryAccessAux<F>>,
    /// Whether `access_aux` contains the complete host log-native replay.
    /// The all-direct CUDA delta route omits this otherwise-unused expansion;
    /// generic finalization must fail rather than assemble a verbose record
    /// when this is false.
    pub access_aux_complete: bool,
    pub to_state: VmState<GuestMemory>,
    pub instret: u64,
    pub suspended: bool,
    /// R3: per migrated chip, the inline compact records the generated C wrote
    /// for this segment. Empty when the library was compiled without inline
    /// records.
    pub inline_records: Vec<RvrInlineChipRecords>,
    /// Stage-2 cross-AIR chronological stream, kept separate so its aligned
    /// page-locked backing can survive decode and return to the right pool.
    pub delta_records: Option<RvrDeltaRecords>,
    /// G2 private compact segment. Its generated-C payload is already in final
    /// wire positions; no host record walk has occurred.
    pub g2_segment: Option<RvrG2SegmentV1>,
    pub g2_meta: Option<Arc<super::RvrG2MetaV1>>,
    /// R3: per program slot, whether that instruction emits an inline compact
    /// record — its memory-log entries are suppressed, so record assembly must
    /// skip the log assembler for it and consume `inline_records` instead.
    pub inline_pc_slots: Arc<Vec<bool>>,
    /// AOT-persisted operand table and whole-AIR classification for the CUDA
    /// delta decoder. This shares the compiled metadata lifetime and is
    /// loaded by pointer identity when the cached VM is constructed.
    pub delta_decode_precomputed: Option<Arc<super::RvrDeltaDecodePrecompute>>,
    /// R4: `(air_idx, written_record_count)` for airs whose records the C
    /// wrote arena-native into caller-provided targets. Record assembly must
    /// skip BOTH the log assembler and the inline assembler for these airs
    /// and only verify the count against the program log.
    pub arena_native_written: Vec<(usize, u32)>,
    /// Exact byte cursor for every arena-native target. Fixed-shape targets
    /// equal `written_count * stride`; packed variable-row targets do not.
    pub arena_native_written_bytes: Vec<(usize, u32)>,
}

pub struct RvrPreflightInstance<'a, F: PrimeField32> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    pub(crate) compiled: RvrCompiled,
    pub(crate) chip_counts_len: usize,
    /// Cross-segment scratch-buffer pool; lives as long as the compiled
    /// library so repeated preflight calls reuse their large allocations.
    pub(crate) pool: RvrPreflightBufferPool,
}

/// Which engine the proving path uses for preflight execution.
///
/// The default is keyed off the prover backend via
/// [`VmBuilder::default_rvr_preflight_engine`](crate::arch::VmBuilder::default_rvr_preflight_engine):
/// CPU provers default to [`Interpreter`](Self::Interpreter), GPU provers to
/// [`Rvr`](Self::Rvr). Rationale (reth block 23992138, 2026-07): the
/// interpreter fuses execute + arena fill in one pass, while the rvr inline
/// path pays a host compact→arena assembly pass that is 55–62% of its
/// preflight time on CPU — 1.80× slower end-to-end at reth scale. On the GPU
/// backend that assembly pass does not exist (compact records are the H2D
/// payload; expansion happens on-device), so rvr remains the default there.
///
/// Overrides, strongest first:
/// 1. [`VmInstance::set_rvr_preflight_engine`](crate::arch::VmInstance::set_rvr_preflight_engine) —
///    programmatic, per instance.
/// 2. `OPENVM_RVR_PREFLIGHT_ENGINE=interpreter|rvr` — environment; any other value panics (loudly,
///    to avoid silently mismeasured benchmarks).
///
/// `Rvr` is a preference, not a guarantee: programs with opcodes outside the
/// rvr preflight surface still fall back to the interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RvrPreflightEngine {
    Interpreter,
    Rvr,
}

/// Reads the `OPENVM_RVR_PREFLIGHT_ENGINE` environment override.
///
/// Panics on an unrecognized value: a typo silently reverting to the default
/// would invalidate any engine A/B measurement built on this knob.
pub fn rvr_preflight_engine_env_override() -> Option<RvrPreflightEngine> {
    match std::env::var("OPENVM_RVR_PREFLIGHT_ENGINE") {
        Ok(value) => match value.as_str() {
            "interpreter" => Some(RvrPreflightEngine::Interpreter),
            "rvr" => Some(RvrPreflightEngine::Rvr),
            other => panic!(
                "OPENVM_RVR_PREFLIGHT_ENGINE must be \"interpreter\" or \"rvr\", got {other:?}"
            ),
        },
        Err(_) => None,
    }
}

pub enum RvrPreflightRoute<'a, F: PrimeField32, E> {
    Rvr(RvrPreflightInstance<'a, F>),
    Interpreter(PreflightInterpretedInstance<F, E>),
}

impl<'a, F, E> RvrPreflightRoute<'a, F, E>
where
    F: PrimeField32,
{
    pub fn is_rvr(&self) -> bool {
        matches!(self, Self::Rvr(_))
    }

    pub fn is_interpreter(&self) -> bool {
        matches!(self, Self::Interpreter(_))
    }
}

impl<'a, F> RvrPreflightInstance<'a, F>
where
    F: PrimeField32,
{
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        exe: Arc<VmExe<F>>,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
        compiled: RvrCompiled,
        chips: &ChipMapping,
    ) -> Self {
        Self {
            system_config,
            exe,
            runtime_hooks,
            compiled,
            chip_counts_len: chip_counts_len(chips),
            pool: RvrPreflightBufferPool::from_env(),
        }
    }

    /// Return a consumed output's large buffers to this instance's pool so
    /// the next segment reuses them instead of re-faulting fresh mappings.
    /// Callers that keep the output alive (differential tests) simply drop it
    /// instead; the pool then refills lazily.
    pub fn recycle_output(&self, output: RvrPreflightOutput<F>) {
        self.pool.recycle_access_aux(output.access_aux);
        #[cfg(feature = "rvr")]
        if let Some(exec_pool) = output.system_records.rvr_exec_frequencies_pool.as_ref() {
            exec_pool.recycle_exec_frequencies(
                output.system_records.filtered_exec_frequencies,
                output.system_records.rvr_exec_frequencies_touched,
            );
            self.pool.recycle_segment_buffers(
                output.raw_logs,
                output.inline_records,
                output.delta_records,
            );
            return;
        }
        self.pool.recycle_segment_buffers(
            output.raw_logs,
            output.inline_records,
            output.delta_records,
        );
    }

    /// Calls [`VmState::initial`] for this fixed executable.
    pub fn create_initial_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        VmState::initial(
            self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        )
    }

    /// Executes this fixed executable from its initial state.
    ///
    /// When `num_insns` is `Some(n)`, `n` must be an rvr basic-block
    /// boundary unless the program naturally terminates before the bound. M2
    /// does not implement exact mid-block suspension; a suspended run that
    /// retires fewer than `n` instructions returns an error instead of
    /// silently producing truncated `SystemRecords`.
    ///
    /// Block-aligned segmentation is complete only when every basic block's
    /// per-chip trace-height contribution fits that chip's max trace height
    /// (`2^log_stacked_height`). Normal compiled RV64IM satisfies this because
    /// branches bound block sizes. A single oversized branchless block cannot
    /// be split at a block boundary and will fail loudly during
    /// commit/aggregation rather than producing a silent invalid proof. Exact
    /// per-instruction suspension would remove this pre-existing rvr metered
    /// segmentation limitation.
    /// The compiled preflight library (compile metadata included), e.g. for
    /// reading `inline_records().arena_native_airs` in tests and callers
    /// that build arena-native record targets.
    pub fn compiled(&self) -> &RvrCompiled {
        &self.compiled
    }

    pub fn execute_preflight(
        &self,
        inputs: impl Into<Streams>,
        num_insns: Option<u64>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        let state = self.create_initial_state(inputs);
        self.execute_preflight_from_state(state, num_insns)
    }

    /// Executes from an already-constructed VM state.
    ///
    /// When `num_insns` is `Some(n)`, `n` must be an rvr basic-block
    /// boundary unless the program naturally terminates before the bound. M2
    /// does not implement exact mid-block suspension; a suspended run that
    /// retires fewer than `n` instructions returns an error instead of
    /// silently producing truncated `SystemRecords`.
    ///
    /// Block-aligned segmentation is complete only when every basic block's
    /// per-chip trace-height contribution fits that chip's max trace height
    /// (`2^log_stacked_height`). Normal compiled RV64IM satisfies this because
    /// branches bound block sizes. A single oversized branchless block cannot
    /// be split at a block boundary and will fail loudly during
    /// commit/aggregation rather than producing a silent invalid proof. Exact
    /// per-instruction suspension would remove this pre-existing rvr metered
    /// segmentation limitation.
    pub fn execute_preflight_from_state(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            &self.exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            None,
            None,
            true,
            false,
            false,
        )
    }

    /// [`Self::execute_preflight_from_state`] with the metered per-AIR trace
    /// heights, taking the single-shot (clone-free, exact-capacity) path the
    /// proving loop uses.
    pub fn execute_preflight_from_state_with_capacities(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: &[u32],
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            &self.exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            Some(record_capacity_rows),
            None,
            true,
            false,
            false,
        )
    }

    /// R4: [`Self::execute_preflight_from_state_with_capacities`] with
    /// caller-provided arena-native record targets. Each `(air ->
    /// ChipRecordBuf)` entry aims that air's records at a pre-allocated arena
    /// backing (stride = row/record pitch, core_off per flavor, base
    /// 32-aligned for Dense adopts); the output reports written byte counts
    /// in `arena_native_written` instead of harvesting bytes.
    pub fn execute_preflight_from_state_with_arena_targets(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: &[u32],
        arena_targets: &BTreeMap<usize, ChipRecordBuf>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            &self.exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            Some(record_capacity_rows),
            Some(arena_targets),
            true,
            false,
            false,
        )
    }

    /// Test-only mirror of the proven all-direct CUDA route: omit host access
    /// aux so generated C emits the compact residual-memory schema into the
    /// arena-native execution fixture.
    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn execute_preflight_from_state_with_compact_delta_memory_for_test(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: &[u32],
        arena_targets: &BTreeMap<usize, ChipRecordBuf>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            &self.exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            Some(record_capacity_rows),
            Some(arena_targets),
            false,
            true,
            false,
        )
    }

    /// Test-only mirror of the complete CUDA ownership signal. Unlike the
    /// compact-schema oracle above, this suppresses host touched-memory
    /// finalization and leaves the chronological inputs for the device.
    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn execute_preflight_from_state_with_device_touched_memory_for_test(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        record_capacity_rows: &[u32],
        arena_targets: &BTreeMap<usize, ChipRecordBuf>,
    ) -> Result<RvrPreflightOutput<F>, ExecutionError> {
        execute_rvr_preflight(
            &self.exe,
            &self.runtime_hooks,
            &self.compiled,
            self.chip_counts_len,
            &self.pool,
            state,
            num_insns,
            Some(record_capacity_rows),
            Some(arena_targets),
            false,
            true,
            true,
        )
    }
}

/// Executes a compiled preflight library. `record_capacity_rows`, when
/// present, bounds each migrated chip's inline-record count for this segment
/// (the metered per-AIR trace heights, indexed by AIR); without it the record
/// buffers fall back to the one-record-per-instruction bound of the program
/// log. `pool` supplies (and receives back) the large per-segment scratch
/// buffers; capacities are still derived per segment exactly as below, so
/// pooling changes allocation lifetime only, never the loud capacity checks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_rvr_preflight<F>(
    exe: &VmExe<F>,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    compiled: &RvrCompiled,
    chip_counts_len: usize,
    pool: &RvrPreflightBufferPool,
    state: VmState<GuestMemory>,
    num_insns: Option<u64>,
    record_capacity_rows: Option<&[u32]>,
    arena_targets: Option<&BTreeMap<usize, ChipRecordBuf>>,
    build_access_aux: bool,
    compact_delta_memory: bool,
    device_touched_memory: bool,
) -> Result<RvrPreflightOutput<F>, ExecutionError>
where
    F: PrimeField32,
{
    assert!(
        arena_targets.is_none() || (num_insns.is_some() && record_capacity_rows.is_some()),
        "arena-native targets require the single-shot proving path"
    );
    assert!(
        !compact_delta_memory
            || ((compiled.inline_records().delta_records
                || compiled.inline_records().g2.is_some())
                && !build_access_aux),
        "compact residual memory requires a fully-direct delta or G2 consumer"
    );
    assert!(
        !device_touched_memory || compact_delta_memory || compiled.inline_records().g2.is_some(),
        "device touched-memory replay requires compact delta or G2 chronology"
    );
    let g2_meta = compiled.inline_records().g2.as_deref();
    let device_aux_oracle = (device_touched_memory
        && std::env::var("OPENVM_RVR_DEVICE_REPLAY_ORACLE").as_deref() == Ok("1"))
        || g2_meta.is_some_and(|meta| meta.checked_emission());
    // A fused-compiled library has NO compact fallback for its arena-native
    // airs: executing without a target would write full records at the
    // compact stride into scratch (garbage). Fail deterministically here
    // instead of tripping the from_pc order guard downstream.
    if let Some(&(air, _)) = compiled
        .inline_records()
        .arena_native_airs
        .iter()
        .find(|(air, _)| !arena_targets.is_some_and(|targets| targets.contains_key(air)))
    {
        return Err(ExecutionError::RvrExecution(format!(
            "air {air} was compiled arena-native but no record target was provided; use the \
             proving path, pass arena targets, or compile with OPENVM_RVR_ARENA_NATIVE=0"
        )));
    }
    let from_state = ExecutionState::new(state.pc(), PREFLIGHT_INITIAL_TIMESTAMP);
    // Per-address-space timestamp-shadow block counts (the C mirror of
    // `TracingMemory.meta`). Blocks are `WORD_SIZE` bytes = `BLOCK_FE_WIDTH`
    // U16 cells, so the block count equals `num_cells / BLOCK_FE_WIDTH`.
    let shadow_blocks = |addr_space: u32| {
        state.memory.memory.config[addr_space as usize]
            .num_cells
            .div_ceil(BLOCK_FE_WIDTH)
            .max(1)
    };
    let reg_shadow_blocks = shadow_blocks(RV64_REGISTER_AS);
    let mem_shadow_blocks = shadow_blocks(RV64_MEMORY_AS);
    let pv_shadow_blocks = shadow_blocks(PUBLIC_VALUES_AS);

    let mut program_log_cap = initial_program_log_cap(exe, num_insns);
    // C1b: on the proving path — exact `num_insns` plus the metered per-AIR
    // trace heights — the log capacities are derived soundly up front, so the
    // run executes directly on `state` with no retry and, critically, no
    // guest-state clone (the dominant per-segment fixed cost: a large flat
    // memory image copy). Soundness of the bound: every memory-log entry
    // belongs to one retired instruction; fixed-shape instructions log well
    // under 64 accesses each (the largest current chips are in the ~30s), and
    // variable-length instructions (multi-word HintStore) log at most 2x their
    // metered trace rows. Overflow is therefore a capacity-model bug and
    // errors loudly below instead of retrying. Callers without metered
    // heights keep the clone-and-retry loop. The buffers are uninitialized
    // (write-only from C), so the generous virtual capacity costs only
    // touched pages.
    let single_shot = num_insns.is_some() && record_capacity_rows.is_some();
    // Single-shot bound: memory-log entries come only from non-migrated
    // instructions, each belonging to a trace row of a NON-inline AIR, and no
    // current chip logs more than 32 accesses per row (largest ~30 for
    // mod-builder rows; multi-word HintStore logs ~1 per row). Inline AIRs
    // are excluded so a migrated-heavy segment does not inflate the
    // reservation (a measured page-table/fault hotspot). Violations error
    // loudly below.
    let mut memory_log_cap = if let (Some(_), Some(heights)) = (num_insns, record_capacity_rows) {
        let inline_meta = compiled.inline_records();
        let non_inline_height_sum: usize = heights
            .iter()
            .enumerate()
            .filter(|(air, _)| {
                !inline_meta
                    .airs
                    .iter()
                    .any(|&(inline_air, _)| inline_air == *air)
            })
            .map(|(_, &h)| h as usize)
            .sum();
        initial_memory_log_cap(program_log_cap).max(non_inline_height_sum.saturating_mul(32) + 64)
    } else {
        initial_memory_log_cap(program_log_cap)
    };
    // The cached proving route selects this only after the builder advertises
    // a fully-direct device delta decoder. Access-aux omission alone is not a
    // discriminator: CPU all-custom arena routes may also safely omit it and
    // must retain the established 40-byte MemoryLogEntry schema.
    let mut state = Some(state);

    loop {
        let mut run_state = if single_shot {
            state.take().expect("single-shot preflight never retries")
        } else {
            state
                .as_ref()
                .expect("retry keeps the pristine state")
                .clone()
        };
        let detailed_profile =
            std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE_DETAIL").as_deref() == Ok("1");
        let native_detailed = std::env::var("OPENVM_RVR_NATIVE_DETAIL").as_deref() == Ok("1");
        let setup_started = std::time::Instant::now();
        // The log buffers are write-only from the C tracer and prefix-read by
        // the host, so they are allocated uninitialized: the zero-fill of
        // multi-GB capacities was a real per-call cost on large segments.
        // They come from (and return to) the cross-segment pool: re-faulting
        // hundreds of MB of fresh mappings per segment was the other measured
        // per-call cost.
        let mut program_log = pool.take_program_log(program_log_cap);
        let mut program_runs =
            pool.take_program_runs(if device_touched_memory && g2_meta.is_none() {
                program_log_cap
            } else {
                0
            });
        let mut device_program_references =
            pool.take_device_program_references(if device_aux_oracle {
                program_log_cap
            } else {
                0
            });
        let mut memory_log = if compact_delta_memory {
            Vec::new()
        } else {
            pool.take_memory_log(memory_log_cap)
        };
        let mut delta_memory_log = if compact_delta_memory {
            pool.take_delta_memory_log(memory_log_cap)
        } else {
            Vec::new()
        };
        let logs_taken = std::time::Instant::now();
        let (
            mut chip_counts,
            mut chip_counts_touched,
            mut exec_frequencies,
            mut exec_frequencies_touched,
        ) = pool.take_counters(chip_counts_len.max(1), || {
            exe.program.num_defined_instructions()
        });
        let counters_ready = std::time::Instant::now();
        // Shadows are all-zero at segment start (0 = untouched this segment);
        // pooled reuse preserves that via the exact touched-list scrub below.
        // Distinct touched blocks are bounded by the number of memory-log
        // entries, so sizing `touched` to `memory_log_cap` guarantees it
        // never overflows.
        let g2_needs_custom_predecessors = g2_meta.is_some();
        let shadow_sizes =
            if device_touched_memory && !device_aux_oracle && !g2_needs_custom_predecessors {
                // Non-null ABI sentinels only: production device aux never indexes
                // a host timestamp shadow.
                (1, 1, 1)
            } else {
                (reg_shadow_blocks, mem_shadow_blocks, pv_shadow_blocks)
            };
        let (mut shadow_register, mut shadow_memory, mut shadow_public_values) =
            pool.take_shadows(shadow_sizes.0, shadow_sizes.1, shadow_sizes.2);
        // Inline (migrated) instructions touch without logging: register
        // touches are bounded by the register file, and each migrated
        // load/store row first-touches at most one memory block, bounded by
        // that chip's metered height (fallback: one per instruction).
        let inline_touch_slack: usize = compiled
            .inline_records()
            .airs
            .iter()
            .map(|&(air, _)| {
                record_capacity_rows
                    .and_then(|heights| heights.get(air))
                    .map(|&height| height as usize)
                    .unwrap_or(program_log_cap)
            })
            .sum::<usize>()
            + 64;
        let touched_cap =
            if device_touched_memory && !device_aux_oracle && !g2_needs_custom_predecessors {
                0
            } else {
                memory_log_cap + inline_touch_slack
            };
        let mut touched = pool.take_touched(touched_cap);
        let pv_base = public_values_slice(&mut run_state.memory.memory).as_mut_ptr();
        let scratch_ready = std::time::Instant::now();

        // R3: per migrated chip, an inline compact-record buffer. Sized by the
        // metered per-AIR trace heights when available, else by one record per
        // instruction.
        let inline_meta = compiled.inline_records();
        // Record buffers are write-only from C (each record fully written,
        // sequential prefix), so they are uninitialized like the logs: the
        // kernel zero-fill of hundreds of MB per call was a measured hotspot.
        let mut record_bufs: Vec<Vec<MaybeUninit<u8>>> = inline_meta
            .airs
            .iter()
            .map(|&(air, record_size)| {
                if inline_meta.delta_records || g2_meta.is_some() {
                    return Vec::new();
                }
                if arena_targets.is_some_and(|targets| targets.contains_key(&air)) {
                    return Vec::new();
                }
                let rows = record_capacity_rows
                    .and_then(|heights| heights.get(air))
                    .map(|&height| height as usize)
                    .unwrap_or(program_log_cap);
                pool.take_record_buf(rows * record_size)
            })
            .collect();
        let record_buffers_ready = std::time::Instant::now();
        let delta_capacity = program_log_cap.saturating_mul(PREFLIGHT_DELTA_RECORD_SIZE);
        let mut delta_output = if inline_meta.delta_records {
            let (backing, needs_prefault) = pool.take_delta_backing(delta_capacity);
            let offset = (32 - backing.as_ptr() as usize % 32) % 32;
            assert!(
                offset + delta_capacity <= backing.len(),
                "aligned delta window exceeds backing"
            );
            Some(RvrDeltaRecords::new(
                backing,
                offset,
                delta_capacity,
                pool.clone(),
                needs_prefault,
            ))
        } else {
            None
        };
        let mut delta_record = ChipRecordBuf::default();
        if let Some(delta) = delta_output.as_mut() {
            let aligned_base = delta.aligned_mut_ptr();
            assert_eq!(
                aligned_base as usize % 32,
                0,
                "delta record base must be 32-byte aligned"
            );
            // A fresh lazy allocation is faulted once, outside the native
            // execute+emit clock. Recycled CPU and pinned CUDA buffers are
            // already resident, so the pool hit skips this page walk.
            if delta.needs_prefault {
                for page in (0..delta_capacity).step_by(4096) {
                    unsafe { std::ptr::write_volatile(aligned_base.add(page), 0) };
                }
                if delta_capacity != 0 {
                    unsafe { std::ptr::write_volatile(aligned_base.add(delta_capacity - 1), 0) };
                }
            }
            delta_record = ChipRecordBuf {
                base: aligned_base.cast(),
                len: 0,
                cap: u32::try_from(delta_capacity).expect("delta capacity exceeds u32"),
                stride: PREFLIGHT_DELTA_RECORD_SIZE as u32,
                core_off: 0,
                flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL
                    | if compact_delta_memory {
                        PREFLIGHT_CHIP_RECORD_FLAG_COMPACT_RESIDUAL_MEMORY
                    } else {
                        0
                    }
                    | if device_touched_memory {
                        PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX
                    } else {
                        0
                    }
                    | if device_aux_oracle {
                        PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_AUX_ORACLE
                    } else {
                        0
                    }
                    | if device_touched_memory {
                        PREFLIGHT_CHIP_RECORD_FLAG_DEVICE_CHRONOLOGY
                    } else {
                        0
                    },
            };
        }
        let delta_ready = std::time::Instant::now();
        let mut g2_prepared = if let Some(g2) = g2_meta {
            let fallback = u32::try_from(program_log_cap)
                .expect("G2 per-kind capacity exceeds the frozen u32 schema");
            let capacities = match (num_insns, record_capacity_rows) {
                (Some(num_insns), Some(trace_heights)) => {
                    RvrG2CapacitiesV1::for_metered_segment(g2, trace_heights, num_insns)?
                }
                _ => {
                    let residual_capacity =
                        g2_residual_capacity(g2, program_log_cap, record_capacity_rows)?;
                    let mut capacities = RvrG2CapacitiesV1 {
                        run: fallback,
                        residual: u32::try_from(residual_capacity)
                            .expect("G2 residual capacity exceeds the frozen u32 schema"),
                        opaque_events: fallback,
                        ..Default::default()
                    };
                    for binding in g2.air_bindings.iter() {
                        capacities.kinds[binding.kind as usize] = record_capacity_rows
                            .and_then(|rows| rows.get(binding.air_idx))
                            .copied()
                            .unwrap_or(fallback);
                    }
                    capacities
                }
            };
            Some(RvrG2PreparedV1::new_pooled_for_mode(
                &capacities,
                pool,
                g2.checked_emission(),
            )?)
        } else {
            None
        };
        let mut chip_records = vec![ChipRecordBuf::default(); chip_counts_len.max(1)];
        for (&(air, record_size), buffer) in inline_meta.airs.iter().zip(record_bufs.iter_mut()) {
            debug_assert!(air < chip_records.len(), "inline air {air} out of range");
            if let Some(target) = arena_targets.and_then(|targets| targets.get(&air)) {
                assert_eq!(target.len, 0, "arena-native target cursor must start at 0");
                assert!(
                    target.stride > 0 && target.cap % target.stride == 0,
                    "arena-native target cap must be a whole number of strides"
                );
                chip_records[air] = *target;
                continue;
            }
            chip_records[air] = ChipRecordBuf {
                base: buffer.as_mut_ptr().cast(),
                len: 0,
                cap: buffer.len() as u32,
                // Compact wire: the stride IS the packed record size. The
                // arena-native mode (R4) instead points `base` into the arena
                // backing and sets the row/record pitch here.
                stride: record_size as u32,
                core_off: 0,
                flags: 0,
            };
        }

        let mut custom_memory_scratch =
            [MaybeUninit::<MemoryLogEntry>::uninit(); PREFLIGHT_CUSTOM_MEMORY_SCRATCH_CAP];
        let device_aux_event_cap = g2_prepared
            .as_ref()
            .map(RvrG2PreparedV1::residual_capacity)
            .unwrap_or(memory_log_cap);
        let device_patch_cap = if device_touched_memory || g2_meta.is_some() {
            device_aux_event_cap
                .checked_mul(2)
                .expect("device aux patch capacity overflow")
        } else {
            0
        };
        let mut device_aux_patches =
            vec![MaybeUninit::<DeviceAuxPatch>::uninit(); device_patch_cap];
        let mut device_aux_references = vec![
            MaybeUninit::<DeviceAuxReference>::uninit();
            if device_aux_oracle {
                device_aux_event_cap
            } else {
                0
            }
        ];
        let memory_bytes = run_state.memory.memory.mem[RV64_MEMORY_AS as usize].size();
        let memory_pages = memory_bytes.div_ceil(PAGE_SIZE);
        let mut dirty_memory_pages = vec![0u64; memory_pages.div_ceil(64)];
        let mut tracer =
            PreflightTracerData::new_uninit(&mut program_log, &mut memory_log, &mut chip_counts);
        if compact_delta_memory {
            tracer.set_delta_memory_log(&mut delta_memory_log);
        }
        tracer.set_shadows(
            &mut shadow_register,
            &mut shadow_memory,
            &mut shadow_public_values,
            pv_base,
            &mut [],
        );
        tracer.set_touched_uninit(&mut touched);
        tracer.set_chip_records(&mut chip_records);
        tracer.set_exec_frequencies(&mut exec_frequencies);
        tracer.set_custom_memory_scratch(&mut custom_memory_scratch);
        if device_touched_memory && g2_meta.is_none() {
            tracer.set_device_aux(
                &mut device_aux_patches,
                &mut device_aux_references,
                &mut dirty_memory_pages,
            );
            tracer.set_device_chronology(&mut program_runs, &mut device_program_references);
        } else if device_touched_memory || g2_meta.is_some() {
            // G2 generated C always marks main-memory writes so the same
            // single-pass producer is valid on CPU oracle/measurement routes
            // and on CUDA. CPU does not publish the bitmap, but still supplies
            // its bounds-checked scratch instead of weakening the native ABI.
            tracer.set_device_aux(
                &mut device_aux_patches,
                &mut device_aux_references,
                &mut dirty_memory_pages,
            );
        }
        tracer.set_counter_touched_uninit(&mut chip_counts_touched, &mut exec_frequencies_touched);
        if inline_meta.delta_records {
            tracer.set_delta_records(&mut delta_record);
        }
        if let Some(g2) = g2_prepared.as_mut() {
            tracer.set_g2(&mut g2.producer);
        }
        let mut native_detail = native_detailed.then(RvrNativeDetail::new);
        if let Some(detail) = native_detail.as_mut() {
            tracer.set_native_detail(detail);
        }

        let setup_finished = std::time::Instant::now();
        let native_execute_started = std::time::Instant::now();
        if let Some(detail) = native_detail.as_mut() {
            detail.start();
        }
        let run_result = execute_preflight_raw(
            compiled,
            runtime_hooks,
            &mut run_state,
            &mut tracer,
            num_insns,
        );
        let native_detail_outer_cycles = native_detail.as_mut().map(RvrNativeDetail::finish);
        let native_execute_elapsed = native_execute_started.elapsed();
        if let (Some(detail), Some(outer_cycles)) =
            (native_detail.as_ref(), native_detail_outer_cycles)
        {
            eprintln!(
                "OPENVM_RVR_NATIVE_DETAIL outer_cycles={outer_cycles} timer_overhead={} \
                 family_cycles={:?} family_instructions={:?} phase_cycles={:?} \
                 phase_samples={:?} phase_events={:?} phase_bytes={:?}",
                detail.timer_overhead,
                detail.family_cycles,
                detail.family_instructions,
                detail.phase_cycles,
                detail.phase_samples,
                detail.phase_events,
                detail.phase_bytes,
            );
        }
        // Drain non-temporal record stores before either consuming the
        // buffers or propagating an execution error. On error, propagation
        // drops the RAII delta lease and may queue its pinned backing for
        // asynchronous cleaning/reuse.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_sfence();
        }
        if let Some(rejection) = g2_prepared
            .as_ref()
            .map(|prepared| prepared.producer.overflow)
            .filter(|&rejection| rejection != 0)
        {
            return Err(ExecutionError::RvrExecution(
                format!(
                    "G2 wire v1: native lane producer rejected a value before publish (code {rejection})"
                ),
            ));
        }
        let mut run_result = run_result.map_err(map_rvr_execute_error)?;
        let post_native_started = std::time::Instant::now();
        if let Some(delta) = delta_output.as_mut() {
            delta.set_written(delta_record.len as usize);
        }
        let mut production_g2_counter_indices = None;
        let g2_segment = if let (Some(prepared), Some(g2)) = (g2_prepared, g2_meta) {
            let mut instruction_count = u32::try_from(run_result.state.mode_state.instret)
                .map_err(|_| {
                    ExecutionError::RvrExecution(
                        "G2 instruction count exceeds the frozen u32 header".to_string(),
                    )
                })?;
            let opaque_written = g2
                .opaque_bindings
                .iter()
                .map(|&binding| {
                    let buf = chip_records.get(binding.air_idx).ok_or_else(|| {
                        ExecutionError::RvrExecution(format!(
                            "G2 opaque AIR {} exceeds the record target table",
                            binding.air_idx
                        ))
                    })?;
                    if buf.stride as usize != binding.geometry.stride_dense()
                        || buf.len % buf.stride != 0
                    {
                        return Err(ExecutionError::RvrExecution(format!(
                            "G2 opaque AIR {} target geometry drifted",
                            binding.air_idx
                        )));
                    }
                    Ok((binding, buf.len / buf.stride, buf.len))
                })
                .collect::<Result<Vec<_>, ExecutionError>>()?;
            if !g2.checked_emission() {
                let (counter_indices, derived_instruction_count) =
                    restore_g2_production_counts(&prepared, g2, &opaque_written, &mut chip_counts)?;
                production_g2_counter_indices = Some(counter_indices);
                instruction_count = derived_instruction_count;
                run_result.state.mode_state.instret = u64::from(derived_instruction_count);
            }
            let expected_kind_counts = g2.checked_emission().then(|| {
                let mut counts = [0u32; 31];
                for binding in g2.air_bindings.iter() {
                    counts[binding.kind as usize] = chip_counts[binding.air_idx];
                }
                counts
            });
            Some(prepared.finalize(
                next_segment_id()?,
                instruction_count,
                expected_kind_counts.as_ref(),
                g2.fingerprint,
                &opaque_written,
            )?)
        } else {
            None
        };
        if std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "OPENVM_RVR_NATIVE_EXEC_EMIT_US={} delta={} delta_records={} delta_bytes={} \
                 memory_records={} touched_blocks={}",
                native_execute_elapsed.as_micros(),
                inline_meta.delta_records as u8,
                delta_record.len / PREFLIGHT_DELTA_RECORD_SIZE as u32,
                delta_record.len,
                tracer.memory_log_len,
                tracer.touched_len,
            );
        }
        if let Some(target_instret) = num_insns {
            if run_result.suspended && run_result.state.mode_state.instret != target_instret {
                return Err(ExecutionError::RvrExecution(format!(
                    "mid-block rvr preflight suspension unsupported: requested num_insns={target_instret}, retired instret={} at an rvr basic-block boundary",
                    run_result.state.mode_state.instret
                )));
            }
        }
        if delta_record.flags & PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW != 0 {
            return Err(ExecutionError::RvrExecution(
                "stage-2 delta record reservation overflow or ABI mismatch".to_string(),
            ));
        }
        if let Some((air, _)) = chip_records
            .iter()
            .enumerate()
            .find(|(_, buf)| buf.flags & PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW != 0)
        {
            return Err(ExecutionError::RvrExecution(format!(
                "arena-native record reservation overflow or ABI mismatch for air {air}"
            )));
        }

        let program_len = tracer.program_log_len as usize;
        let program_runs_len = tracer.program_runs_len as usize;
        let program_instruction_len = tracer.program_instruction_len as usize;
        let device_program_references_len = tracer.device_program_references_len as usize;
        let memory_len = tracer.memory_log_len as usize;
        let device_aux_event_count = if let Some(segment) = g2_segment.as_ref() {
            segment.header_acquire()?.residual_event_count as usize
        } else {
            memory_len
        };
        let touched_len = tracer.touched_len as usize;
        let device_aux_patches_len = tracer.device_aux_patches_len as usize;
        let chip_counts_touched_len = tracer.chip_counts_touched_len as usize;
        let exec_frequencies_touched_len = tracer.exec_frequencies_touched_len as usize;
        if chip_counts_touched_len > chip_counts_touched.len()
            || exec_frequencies_touched_len > exec_frequencies_touched.len()
        {
            return Err(ExecutionError::RvrExecution(format!(
                "preflight counter first-touch overflow: chip {chip_counts_touched_len}/{}, \
                 execution {exec_frequencies_touched_len}/{}",
                chip_counts_touched.len(),
                exec_frequencies_touched.len(),
            )));
        }
        // SAFETY: generated C writes an index before advancing each cursor;
        // the capacity check above rejects any cursor outside the buffers.
        let mut chip_counts_touched =
            unsafe { assume_init_prefix(chip_counts_touched, chip_counts_touched_len) };
        if let Some(indices) = production_g2_counter_indices {
            chip_counts_touched.clear();
            chip_counts_touched.extend(
                indices
                    .into_iter()
                    .filter(|&air_idx| chip_counts[air_idx] != 0)
                    .map(|air_idx| air_idx as u32),
            );
        }
        let exec_frequencies_touched =
            unsafe { assume_init_prefix(exec_frequencies_touched, exec_frequencies_touched_len) };
        // With inline records (R3), migrated opcodes touch without logging, so
        // `touched` can outgrow the memory log; growing `memory_log_cap` grows
        // the touched buffer with it.
        if program_len > program_log_cap
            || memory_len > memory_log_cap
            || touched_len > touched.len()
            || device_aux_patches_len > device_aux_patches.len()
            || program_runs_len > program_runs.len()
            || program_instruction_len > program_log_cap
            || device_program_references_len > device_program_references.len()
            || delta_record.len as usize > delta_capacity
        {
            if single_shot {
                // The metered-derived bound was violated: a capacity-model
                // bug (a chip logging more accesses per instruction/row than
                // the bound assumes), not a runtime condition to retry.
                return Err(ExecutionError::RvrExecution(format!(
                    "metered-derived preflight log capacity exceeded: \
                     program {program_len}/{program_log_cap}, \
                     memory {memory_len}/{memory_log_cap}, \
                     touched {touched_len}/{}, runs {program_runs_len}/{}, chronology \
                     {program_instruction_len}/{program_log_cap}, delta {}/{} — capacity-model bug",
                    touched.len(),
                    program_runs.len(),
                    delta_record.len,
                    delta_capacity
                )));
            }
            program_log_cap = grow_capacity(program_log_cap, program_len);
            memory_log_cap = grow_capacity(memory_log_cap, memory_len.max(touched_len));
            // Recycle for the grown retry: the log/touched buffers go back as
            // uninit spares (content irrelevant; any spare smaller than the
            // next take is dropped there), and the record buffers are
            // height-sized and still fit. The shadows are dirty and the
            // touched list that would scrub them may itself have overflowed,
            // so they drop here and the retry takes fresh zeroed ones.
            pool.recycle_raw_uninit(
                program_log,
                program_runs,
                device_program_references,
                memory_log,
                delta_memory_log,
                touched,
            );
            pool.recycle_chip_counts(chip_counts, chip_counts_touched);
            pool.recycle_exec_frequencies(exec_frequencies, exec_frequencies_touched);
            for buf in record_bufs {
                pool.recycle_record_buf(buf);
            }
            continue;
        }

        // SAFETY: the C tracer fully wrote the first `*_len` entries of each
        // log (its append helpers bounds-check against the caps, and the
        // overflow check above rejected any run whose counters passed them).
        let program_log = unsafe { assume_init_prefix(program_log, program_len) };
        let program_runs = unsafe { assume_init_prefix(program_runs, program_runs_len) };
        let device_program_references =
            unsafe { assume_init_prefix(device_program_references, device_program_references_len) };
        if device_touched_memory && g2_meta.is_none() {
            if program_instruction_len != run_result.state.mode_state.instret as usize {
                return Err(ExecutionError::RvrExecution(format!(
                    "device chronology covered {program_instruction_len} instructions but native execution retired {}",
                    run_result.state.mode_state.instret
                )));
            }
            if program_instruction_len != 0 && program_runs.is_empty() {
                return Err(ExecutionError::RvrExecution(
                    "non-empty device chronology emitted no block runs".to_string(),
                ));
            }
            if let Some((index, run)) = program_runs
                .iter()
                .enumerate()
                .find(|(_, run)| run.complete != 1)
            {
                return Err(ExecutionError::RvrExecution(format!(
                    "device chronology run {index} is incomplete ({})",
                    run.complete
                )));
            }
            if device_aux_oracle && device_program_references.len() != program_instruction_len {
                return Err(ExecutionError::RvrExecution(format!(
                    "device chronology oracle retained {} instructions for {program_instruction_len} reconstructed entries",
                    device_program_references.len()
                )));
            }
        }
        let memory_log = if compact_delta_memory {
            Vec::new()
        } else {
            unsafe { assume_init_prefix(memory_log, memory_len) }
        };
        let delta_memory_log = if compact_delta_memory {
            unsafe { assume_init_prefix(delta_memory_log, memory_len) }
        } else {
            Vec::new()
        };
        let touched = unsafe { assume_init_prefix(touched, touched_len) };
        // SAFETY: each custom predecessor store writes a complete descriptor
        // before incrementing no further than the checked C-side capacity.
        let device_aux_patches =
            unsafe { assume_init_prefix(device_aux_patches, device_aux_patches_len) };
        let device_aux_references = if device_aux_oracle {
            // The oracle writes one complete reference per compact residual
            // event at the same event index.
            unsafe { assume_init_prefix(device_aux_references, device_aux_event_count) }
        } else {
            Vec::new()
        };
        if !device_aux_patches.is_empty() {
            let targets = arena_targets.ok_or_else(|| {
                ExecutionError::RvrExecution(
                    "device aux patches require live staged arenas".to_string(),
                )
            })?;
            let mut target_ranges = Vec::with_capacity(device_aux_patches.len());
            for (patch_index, patch) in device_aux_patches.iter().enumerate() {
                let width = match patch.kind {
                    DEVICE_AUX_PATCH_U32 => 4usize,
                    DEVICE_AUX_PATCH_U64 => 8usize,
                    kind => {
                        return Err(ExecutionError::RvrExecution(format!(
                            "device aux patch {patch_index} has invalid kind {kind}"
                        )));
                    }
                };
                if patch.event_index as usize >= device_aux_event_count {
                    return Err(ExecutionError::RvrExecution(format!(
                        "device aux patch {patch_index} references residual event {} of {device_aux_event_count}",
                        patch.event_index,
                    )));
                }
                let target = patch.target as usize;
                let target_end = target.checked_add(width).ok_or_else(|| {
                    ExecutionError::RvrExecution(format!(
                        "device aux patch {patch_index} target range overflows"
                    ))
                })?;
                let owners = targets
                    .iter()
                    .filter_map(|(&air_idx, arena)| {
                        let begin = arena.base as usize;
                        let written = chip_records.get(air_idx)?.len as usize;
                        let end = begin.checked_add(written)?;
                        (target >= begin && target_end <= end).then_some(air_idx)
                    })
                    .collect::<Vec<_>>();
                if owners.len() != 1 {
                    return Err(ExecutionError::RvrExecution(format!(
                        "device aux patch {patch_index} target {target:#x} has {} written-arena owners",
                        owners.len(),
                    )));
                }
                target_ranges.push((target, target_end, patch_index, owners[0]));
            }
            target_ranges.sort_unstable_by_key(|&(begin, _, _, _)| begin);
            if let Some(pair) = target_ranges.windows(2).find(|pair| pair[0].1 > pair[1].0) {
                return Err(ExecutionError::RvrExecution(format!(
                    "device aux patches {} (AIR {}) and {} (AIR {}) overlap",
                    pair[0].2, pair[0].3, pair[1].2, pair[1].3,
                )));
            }
        }
        let logs_ready = std::time::Instant::now();

        let shadows = PreflightShadowsView {
            register: &shadow_register,
            memory: &shadow_memory,
            public_values: &shadow_public_values,
        };
        let replay = if device_touched_memory && !device_aux_oracle {
            // The all-direct CUDA route already replays these chronological
            // inputs. It also emits the canonical sorted final touched blocks,
            // so avoid repeating the O(touched) collect + sort on the host.
            PreflightMemoryReplay {
                touched_memory: Vec::new(),
                access_aux: Vec::new(),
            }
        } else {
            let replay_fusion = pool.replay_fusion_enabled();
            let access_aux_backing = (replay_fusion && build_access_aux)
                .then(|| pool.take_access_aux::<F>(memory_log.len()));
            let mut touched_order = replay_fusion.then(|| {
                pool.take_touched_order(
                    shadow_register.len(),
                    shadow_memory.len(),
                    shadow_public_values.len(),
                )
            });
            let replay = build_preflight_replay_with_scratch::<F>(
                &run_state.memory,
                &shadows,
                &touched,
                &memory_log,
                build_access_aux,
                access_aux_backing,
                touched_order.as_mut(),
            );
            if let Some(touched_order) = touched_order {
                pool.recycle_touched_order(touched_order);
            }
            replay.map_err(|err| ExecutionError::RvrExecution(err.to_string()))?
        };
        let replay_ready = std::time::Instant::now();
        if device_touched_memory && device_aux_oracle {
            extend_touched_pages_from_raw(&mut run_state.memory, &touched);
        } else if device_touched_memory {
            // Sparse continuation transport needs only dirty pages, not the
            // first-touch set. Writes mark this tiny bitmap in the native loop
            // without timestamp shadows or address-to-block translation.
            let pages = &mut run_state.memory.memory.touched_pages[RV64_MEMORY_AS as usize];
            for (word_index, &word) in dirty_memory_pages.iter().enumerate() {
                let mut pending = word;
                while pending != 0 {
                    let bit = pending.trailing_zeros() as usize;
                    let page = word_index * 64 + bit;
                    pages.mark_byte_range(page * PAGE_SIZE, 1);
                    pending &= pending - 1;
                }
            }
            for addr_space in [RV64_REGISTER_AS, PUBLIC_VALUES_AS] {
                let index = addr_space as usize;
                let size = run_state.memory.memory.mem[index].size();
                if size != 0 {
                    // Both spaces are small in production (registers are one
                    // page; public values are normally tens of bytes). Mark
                    // the complete space so non-default larger public-value
                    // configurations remain continuation-safe without a
                    // second per-address-space native dirty bitmap.
                    run_state.memory.memory.touched_pages[index].mark_byte_range(0, size);
                }
            }
        } else {
            run_state
                .memory
                .memory
                .extend_touched_pages_from_touched(&replay.touched_memory);
        }
        let touched_pages_ready = std::time::Instant::now();
        // The shadows and the touched list are segment-lifetime scratch: scrub
        // the shadows back to all-zero via the touched list (exact — every
        // nonzero slot was recorded once on its 0→nonzero transition, and the
        // overflow check above guarantees the list is complete) and return
        // both to the pool. The escaping log/record buffers round-trip later
        // through `recycle_segment_buffers` at the consumption seam.
        scrub_shadows(
            &mut shadow_register,
            &mut shadow_memory,
            &mut shadow_public_values,
            &touched,
        );
        pool.prepare_shadow_locality(&mut shadow_memory, &touched);
        pool.recycle_shadows(shadow_register, shadow_memory, shadow_public_values);
        let touched = if inline_meta.delta_records || (g2_meta.is_some() && device_aux_oracle) {
            touched
        } else {
            pool.recycle_touched(touched);
            Vec::new()
        };
        let scratch_recycled = std::time::Instant::now();
        let filtered_exec_frequencies = exec_frequencies;
        let to_state = ExecutionState::new(run_state.pc(), tracer.timestamp);
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code: run_result.exit_code,
            filtered_exec_frequencies,
            program_frequencies_on_device: device_touched_memory,
            touched_memory: replay.touched_memory,
            touched_memory_on_device: device_touched_memory,
            device_replay_oracle: device_touched_memory && device_aux_oracle,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_touched: exec_frequencies_touched,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_pool: Some(pool.clone()),
        };

        // R3: harvest each migrated chip's inline records (the C-advanced
        // cursor gives the written length).
        let arena_native_written: Vec<(usize, u32)> = inline_meta
            .airs
            .iter()
            .filter(|&&(air_idx, _)| {
                arena_targets.is_some_and(|targets| targets.contains_key(&air_idx))
            })
            .map(|&(air_idx, _)| {
                let buf = &chip_records[air_idx];
                if buf.flags & PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS != 0 {
                    assert_eq!(
                        buf.core_off, chip_counts[air_idx],
                        "variable-row arena count for air {air_idx} must match metered chip rows"
                    );
                    (air_idx, buf.core_off)
                } else {
                    assert_eq!(
                        buf.len % buf.stride,
                        0,
                        "arena-native cursor for air {air_idx} is not a whole record count"
                    );
                    (air_idx, buf.len / buf.stride)
                }
            })
            .collect();
        let arena_native_written_bytes: Vec<(usize, u32)> = inline_meta
            .airs
            .iter()
            .filter(|&&(air_idx, _)| {
                arena_targets.is_some_and(|targets| targets.contains_key(&air_idx))
            })
            .map(|&(air_idx, _)| (air_idx, chip_records[air_idx].len))
            .collect();
        let device_aux_arena_references =
            if device_aux_oracle && !arena_native_written_bytes.is_empty() {
                let targets = arena_targets.expect("device replay oracle requires staged arenas");
                arena_native_written_bytes
                    .iter()
                    .map(|&(air_idx, written)| {
                        let target = targets
                            .get(&air_idx)
                            .expect("arena-native oracle target disappeared");
                        let written = written as usize;
                        assert!(
                            written <= target.cap as usize,
                            "arena-native oracle cursor exceeds target for air {air_idx}"
                        );
                        // SAFETY: the target is a live staged arena allocation,
                        // and the native writer cap-checked and initialized this
                        // complete prefix before returning.
                        let expected = unsafe {
                            std::slice::from_raw_parts(target.base.cast_const(), written).to_vec()
                        };
                        DeviceAuxArenaReference {
                            air_idx,
                            base: target.base as u64,
                            expected,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
        let inline_records: Vec<RvrInlineChipRecords> =
            if inline_meta.delta_records || g2_meta.is_some() {
                Vec::new()
            } else {
                inline_meta
                    .airs
                    .iter()
                    .zip(record_bufs.iter_mut())
                    .map(|(&(air_idx, record_size), buffer)| {
                        if arena_targets.is_some_and(|targets| targets.contains_key(&air_idx)) {
                            return RvrInlineChipRecords {
                                air_idx,
                                record_size,
                                bytes: Vec::new(),
                            };
                        }
                        let written = chip_records[air_idx].len as usize;
                        // SAFETY: the C tracer fully writes each emitted record and
                        // advances the cursor past it, so the first `written` bytes
                        // are initialized (cap-checked on the C side).
                        let bytes = unsafe { assume_init_prefix(std::mem::take(buffer), written) };
                        RvrInlineChipRecords {
                            air_idx,
                            record_size,
                            bytes,
                        }
                    })
                    .collect()
            };
        let delta_records = delta_output;
        let harvest_ready = std::time::Instant::now();

        if detailed_profile {
            eprintln!(
                "OPENVM_RVR_PREFLIGHT_DETAIL setup_us={} log_take_us={} counter_alloc_us={} \
                 scratch_take_us={} record_buffer_us={} delta_take_us={} descriptor_us={} \
                 native_us={} checks_us={} replay_us={} touched_pages_us={} scrub_recycle_us={} \
                 harvest_us={} program_records={} memory_records={} touched_blocks={} \
                 program_log_bytes={} memory_log_bytes={} touched_log_bytes={} \
                 arena_native_airs={} access_aux_required={}",
                (setup_finished - setup_started).as_micros(),
                (logs_taken - setup_started).as_micros(),
                (counters_ready - logs_taken).as_micros(),
                (scratch_ready - counters_ready).as_micros(),
                (record_buffers_ready - scratch_ready).as_micros(),
                (delta_ready - record_buffers_ready).as_micros(),
                (setup_finished - delta_ready).as_micros(),
                native_execute_elapsed.as_micros(),
                (logs_ready - post_native_started).as_micros(),
                (replay_ready - logs_ready).as_micros(),
                (touched_pages_ready - replay_ready).as_micros(),
                (scratch_recycled - touched_pages_ready).as_micros(),
                (harvest_ready - scratch_recycled).as_micros(),
                program_len,
                memory_len,
                touched_len,
                program_len * std::mem::size_of::<ProgramLogEntry>(),
                memory_len
                    * if compact_delta_memory {
                        std::mem::size_of::<DeltaMemoryLogEntry>()
                    } else {
                        std::mem::size_of::<MemoryLogEntry>()
                    },
                touched_len * std::mem::size_of::<TouchedBlock>(),
                arena_native_written.len(),
                build_access_aux as u8,
            );
        }

        return Ok(RvrPreflightOutput {
            system_records,
            raw_logs: PreflightRawLogs {
                program_log,
                program_runs,
                device_program_references,
                memory_log,
                delta_memory_log,
                chip_counts,
                chip_counts_touched,
                touched,
                device_aux_patches,
                device_aux_references,
                device_aux_arena_references,
            },
            access_aux: replay.access_aux,
            access_aux_complete: build_access_aux && g2_meta.is_none(),
            to_state: run_state,
            instret: run_result.state.mode_state.instret,
            suspended: run_result.suspended,
            inline_records,
            delta_records,
            g2_segment,
            g2_meta: inline_meta.g2.clone(),
            inline_pc_slots: inline_meta.pc_slots.clone(),
            delta_decode_precomputed: inline_meta.delta_decode.clone(),
            arena_native_written,
            arena_native_written_bytes,
        });
    }
}

fn extend_touched_pages_from_raw(memory: &mut GuestMemory, touched: &[TouchedBlock]) {
    for block in touched {
        memory.memory.touched_pages[block.addr_space as usize]
            .mark_byte_range(block.block_addr as usize, WORD_BYTES);
    }
}

fn chip_counts_len(chips: &ChipMapping) -> usize {
    chips
        .pc_to_chip
        .iter()
        .filter_map(|chip| match chip {
            TraceChipIndex::Chip(air_idx) => Some(air_idx.as_u32() as usize + 1),
            TraceChipIndex::NoChip => None,
        })
        .max()
        .unwrap_or(0)
}

fn initial_program_log_cap<F: Field>(exe: &VmExe<F>, num_insns: Option<u64>) -> usize {
    let expected = num_insns
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or_else(|| exe.program.num_defined_instructions().max(1));
    expected.saturating_add(16).max(64)
}

fn initial_memory_log_cap(program_log_cap: usize) -> usize {
    program_log_cap
        .saturating_mul(8)
        .saturating_add(64)
        .max(128)
}

fn restore_g2_production_counts(
    prepared: &RvrG2PreparedV1,
    meta: &super::RvrG2MetaV1,
    opaque_written: &[(super::RvrG2OpaqueBindingV1, u32, u32)],
    chip_counts: &mut [u32],
) -> Result<(Vec<usize>, u32), ExecutionError> {
    let mut owned_airs = meta
        .air_bindings
        .iter()
        .map(|binding| binding.air_idx)
        .chain(meta.opaque_bindings.iter().map(|binding| binding.air_idx))
        .collect::<Vec<_>>();
    owned_airs.sort_unstable();
    owned_airs.dedup();
    if owned_airs
        .iter()
        .any(|&air_idx| air_idx >= chip_counts.len())
    {
        return Err(ExecutionError::RvrExecution(
            "G2 production AIR exceeds the chip-count table".to_string(),
        ));
    }

    // HintBuffer's custom emitter contributes only its dynamic (n - 1)
    // correction. Preserve that rare value while replacing every per-block
    // base count with lane/run-derived totals outside the generated hot loop.
    let hint_dynamic_extra = meta.air_idx(30).map_or(0, |air_idx| chip_counts[air_idx]);
    for &air_idx in &owned_airs {
        chip_counts[air_idx] = 0;
    }

    for binding in meta.air_bindings.iter().filter(|binding| binding.kind < 30) {
        if let Some(slot) =
            rvr_openvm_ext_ffi_common::g2_standard_producer_slot(binding.kind, false)
        {
            let count = prepared.producer_lane_len(slot)?;
            chip_counts[binding.air_idx] = chip_counts[binding.air_idx]
                .checked_add(count)
                .ok_or_else(|| {
                    ExecutionError::RvrExecution(
                        "G2 production standard chip count overflow".to_string(),
                    )
                })?;
        }
    }

    if meta.blocks.len() != meta.block_host_counts.len() {
        return Err(ExecutionError::RvrExecution(
            "G2 host block-count table drifted from the device table".to_string(),
        ));
    }
    let mut host_counts = super::RvrG2BlockHostCountsV1::default();
    let mut instruction_count = 0u32;
    for &program_slot in
        prepared.producer_u32_lane(rvr_openvm_ext_ffi_common::G2_PRODUCER_RUN_SLOT)?
    {
        let block_index = meta
            .blocks
            .binary_search_by_key(&program_slot, |block| block.program_slot)
            .map_err(|_| {
                ExecutionError::RvrExecution(format!(
                    "G2 production run references unknown block slot {program_slot}"
                ))
            })?;
        let block_entry = meta.blocks[block_index];
        instruction_count = instruction_count
            .checked_add(block_entry.instruction_count)
            .ok_or_else(|| {
                ExecutionError::RvrExecution("G2 production instruction count overflow".to_string())
            })?;
        let block = meta.block_host_counts[block_index];
        host_counts.kind12 = host_counts
            .kind12
            .checked_add(block.kind12)
            .ok_or_else(|| ExecutionError::RvrExecution("G2 kind 12 count overflow".to_string()))?;
        host_counts.kind14 = host_counts
            .kind14
            .checked_add(block.kind14)
            .ok_or_else(|| ExecutionError::RvrExecution("G2 kind 14 count overflow".to_string()))?;
        host_counts.kind30 = host_counts
            .kind30
            .checked_add(block.kind30)
            .ok_or_else(|| ExecutionError::RvrExecution("G2 kind 30 count overflow".to_string()))?;
    }
    for (kind, count) in [
        (12, host_counts.kind12),
        (14, host_counts.kind14),
        (30, host_counts.kind30),
    ] {
        if let Some(air_idx) = meta.air_idx(kind) {
            chip_counts[air_idx] = chip_counts[air_idx].checked_add(count).ok_or_else(|| {
                ExecutionError::RvrExecution(format!(
                    "G2 production kind {kind} chip count overflow"
                ))
            })?;
        } else if count != 0 {
            return Err(ExecutionError::RvrExecution(format!(
                "G2 production counted unbound kind {kind}"
            )));
        }
    }
    if let Some(air_idx) = meta.air_idx(30) {
        chip_counts[air_idx] = chip_counts[air_idx]
            .checked_add(hint_dynamic_extra)
            .ok_or_else(|| {
                ExecutionError::RvrExecution("G2 dynamic hint count overflow".to_string())
            })?;
    } else if hint_dynamic_extra != 0 {
        return Err(ExecutionError::RvrExecution(
            "G2 dynamic hint count has no AIR binding".to_string(),
        ));
    }

    for &(binding, count, _) in opaque_written {
        chip_counts[binding.air_idx] = count;
    }
    Ok((owned_airs, instruction_count))
}

fn g2_residual_capacity(
    meta: &super::RvrG2MetaV1,
    program_log_cap: usize,
    record_capacity_rows: Option<&[u32]>,
) -> Result<usize, ExecutionError> {
    // Every standard crossing emits at most two residual block events. The
    // HintStore decoder emits at most three events per metered row. Opaque
    // extensions publish an exact per-record maximum with their frozen arena
    // geometry, so a custom-heavy executable is not constrained by the
    // legacy eight-events-per-instruction raw-log estimate.
    let mut capacity = program_log_cap
        .checked_mul(2)
        .and_then(|value| value.checked_add(64))
        .ok_or_else(|| ExecutionError::RvrExecution("G2 residual capacity overflow".to_string()))?;
    if let Some(binding) = meta.air_bindings.iter().find(|binding| binding.kind == 30) {
        let rows = record_capacity_rows
            .and_then(|heights| heights.get(binding.air_idx))
            .map_or(program_log_cap, |&height| height as usize);
        capacity = capacity
            .checked_add(rows.checked_mul(3).ok_or_else(|| {
                ExecutionError::RvrExecution("G2 HintStore capacity overflow".to_string())
            })?)
            .ok_or_else(|| {
                ExecutionError::RvrExecution("G2 residual capacity overflow".to_string())
            })?;
    }
    for binding in meta.opaque_bindings.iter() {
        let rows = record_capacity_rows
            .and_then(|heights| heights.get(binding.air_idx))
            .map_or(program_log_cap, |&height| height as usize);
        capacity = capacity
            .checked_add(
                rows.checked_mul(binding.max_residual_events_per_record as usize)
                    .ok_or_else(|| {
                        ExecutionError::RvrExecution(
                            "G2 opaque residual capacity overflow".to_string(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                ExecutionError::RvrExecution("G2 residual capacity overflow".to_string())
            })?;
    }
    let capacity = capacity.max(128);
    if capacity >= DEVICE_AUX_EVENT_INDEX_LIMIT {
        return Err(ExecutionError::RvrExecution(
            "G2 residual capacity exceeds device aux token space".to_string(),
        ));
    }
    Ok(capacity)
}

fn grow_capacity(current: usize, needed: usize) -> usize {
    current
        .saturating_mul(2)
        .max(needed.saturating_mul(2))
        .max(1)
}

/// Converts the written prefix of an externally filled uninit buffer into an
/// initialized `Vec` without copying.
///
/// # Safety
/// The first `len` elements must have been fully written.
unsafe fn assume_init_prefix<T>(mut buffer: Vec<MaybeUninit<T>>, len: usize) -> Vec<T> {
    assert!(len <= buffer.len());
    let ptr = buffer.as_mut_ptr().cast::<T>();
    let cap = buffer.capacity();
    std::mem::forget(buffer);
    // SAFETY: `MaybeUninit<T>` and `T` have identical layout; the caller
    // guarantees the first `len` elements are initialized.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use std::collections::HashSet;

    use openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, SysPhantom, SystemOpcode, VmOpcode, DEFERRAL_AS,
        PUBLIC_VALUES_AS as AS_PUBLIC_VALUES,
    };
    use p3_baby_bear::BabyBear;
    use rvr_openvm_lift::{AirIndex, TraceChipIndex};

    use super::*;
    use crate::{
        arch::{
            rvr::{
                classify_preflight_opcodes, compile_preflight, execute::execute_preflight_for_test,
                ChipMapping, RvrPreflightOpcodeClass,
            },
            Streams, VmState,
        },
        system::memory::{online::PAGE_SIZE, AddressMap, TimestampedValues},
        utils::test_system_config,
    };

    #[test]
    fn raw_first_touches_mark_the_same_continuation_pages_as_finalized_records() {
        let config = test_system_config();
        let raw_touched = [
            TouchedBlock {
                addr_space: RV64_REGISTER_AS,
                block_addr: 8,
                initial_value: 0,
            },
            TouchedBlock {
                addr_space: RV64_MEMORY_AS,
                block_addr: (PAGE_SIZE - WORD_BYTES) as u32,
                initial_value: 0,
            },
            TouchedBlock {
                addr_space: RV64_MEMORY_AS,
                block_addr: (3 * PAGE_SIZE + 8) as u32,
                initial_value: 0,
            },
        ];
        let mut raw_memory = GuestMemory::new(AddressMap::from_mem_config(&config.memory_config));
        let mut finalized_memory = raw_memory.clone();
        extend_touched_pages_from_raw(&mut raw_memory, &raw_touched);

        let finalized = raw_touched
            .iter()
            .map(|block| {
                (
                    (block.addr_space, block.block_addr / 2),
                    TimestampedValues {
                        timestamp: 1,
                        values: [BabyBear::default(); BLOCK_FE_WIDTH],
                    },
                )
            })
            .collect::<Vec<_>>();
        finalized_memory
            .memory
            .extend_touched_pages_from_touched(&finalized);

        for addr_space in [RV64_REGISTER_AS, RV64_MEMORY_AS] {
            let total_bytes = raw_memory.memory.config[addr_space as usize].size();
            assert_eq!(
                raw_memory.memory.touched_pages[addr_space as usize]
                    .touched_byte_ranges(total_bytes),
                finalized_memory.memory.touched_pages[addr_space as usize]
                    .touched_byte_ranges(total_bytes),
                "raw and finalized touched blocks must stage identical continuation pages for AS {addr_space}"
            );
        }
    }

    /// Owns the per-address-space timestamp shadows + touched buffer for a
    /// direct-`execute_preflight_for_test` call and attaches them to the tracer.
    struct TestShadows {
        register: Vec<u32>,
        memory: Vec<u32>,
        public_values: Vec<u32>,
        touched: Vec<TouchedBlock>,
    }

    impl TestShadows {
        fn new(vm_state: &VmState, touched_cap: usize) -> Self {
            let blocks = |addr_space: u32| {
                vm_state.memory.memory.config[addr_space as usize]
                    .num_cells
                    .div_ceil(BLOCK_FE_WIDTH)
                    .max(1)
            };
            Self {
                register: vec![0u32; blocks(RV64_REGISTER_AS)],
                memory: vec![0u32; blocks(RV64_MEMORY_AS)],
                public_values: vec![0u32; blocks(AS_PUBLIC_VALUES)],
                touched: vec![TouchedBlock::default(); touched_cap],
            }
        }

        fn attach(&mut self, vm_state: &mut VmState, tracer: &mut PreflightTracerData) {
            let pv_base = public_values_slice(&mut vm_state.memory.memory).as_mut_ptr();
            tracer.set_shadows(
                &mut self.register,
                &mut self.memory,
                &mut self.public_values,
                pv_base,
                &mut self.touched,
            );
        }
    }

    const OPCODE_ADDI: usize = 0x290;
    const OPCODE_LOADD: usize = 0x210;
    const OPCODE_LOADBU: usize = 0x211;
    const OPCODE_LOADHU: usize = 0x212;
    const OPCODE_LOADWU: usize = 0x213;
    const OPCODE_STORED: usize = 0x214;
    const OPCODE_STOREW: usize = 0x215;
    const OPCODE_STOREH: usize = 0x216;
    const OPCODE_STOREB: usize = 0x217;
    const OPCODE_LOADB: usize = 0x218;
    const OPCODE_LOADH: usize = 0x219;
    const OPCODE_LOADW: usize = 0x21a;
    const TEST_CHIP: u32 = 0;
    const PHANTOM_CHIP: u32 = 1;

    fn reg(idx: usize) -> usize {
        idx * RV64_REGISTER_NUM_LIMBS
    }

    fn addi(rd: usize, rs1: usize, imm: usize) -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(OPCODE_ADDI),
            [reg(rd), reg(rs1), imm, 1, 0],
        )
    }

    fn load_d(rd: usize, rs1: usize, offset: usize) -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(OPCODE_LOADD),
            [reg(rd), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
        )
    }

    fn load_width(opcode: usize, rd: usize, rs1: usize, offset: usize) -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(opcode),
            [reg(rd), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
        )
    }

    fn store_d(rs2: usize, rs1: usize, offset: usize) -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(OPCODE_STORED),
            [reg(rs2), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
        )
    }

    fn store_width(opcode: usize, rs2: usize, rs1: usize, offset: usize) -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(opcode),
            [reg(rs2), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
        )
    }

    fn reveal_like_store() -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(OPCODE_STORED),
            [reg(1), reg(0), 0, 1, AS_PUBLIC_VALUES as usize, 1, 0],
        )
    }

    fn deferral_like_store() -> Instruction<BabyBear> {
        Instruction::from_usize(
            VmOpcode::from_usize(OPCODE_STORED),
            [reg(1), reg(0), 0, 1, DEFERRAL_AS as usize, 1, 0],
        )
    }

    fn terminate() -> Instruction<BabyBear> {
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0)
    }

    fn phantom(sys: SysPhantom) -> Instruction<BabyBear> {
        Instruction::from_isize(
            SystemOpcode::PHANTOM.global_opcode(),
            0,
            0,
            sys as isize,
            0,
            0,
        )
    }

    fn rv64im_memory_exe() -> VmExe<BabyBear> {
        let instructions = [
            addi(1, 0, 64),
            addi(2, 0, 7),
            store_d(2, 1, 0),
            load_d(3, 1, 0),
            addi(4, 0, 0x5a),
            store_width(OPCODE_STOREB, 4, 1, 1),
            load_width(OPCODE_LOADBU, 5, 1, 1),
            store_width(OPCODE_STOREH, 4, 1, 2),
            load_width(OPCODE_LOADHU, 6, 1, 2),
            store_width(OPCODE_STOREW, 4, 1, 4),
            load_width(OPCODE_LOADWU, 7, 1, 4),
            load_width(OPCODE_LOADB, 8, 1, 1),
            load_width(OPCODE_LOADH, 9, 1, 2),
            load_width(OPCODE_LOADW, 10, 1, 4),
            load_d(0, 1, 0),
            terminate(),
        ];
        VmExe::new(Program::from_instructions(&instructions))
    }

    fn phantom_timestamp_exe() -> VmExe<BabyBear> {
        let instructions = [
            addi(1, 0, 64),
            phantom(SysPhantom::Nop),
            addi(2, 0, 7),
            phantom(SysPhantom::CtStart),
            store_d(2, 1, 0),
            phantom(SysPhantom::CtEnd),
            load_d(3, 1, 0),
            terminate(),
        ];
        VmExe::new(Program::from_instructions(&instructions))
    }

    fn chip_mapping(exe: &VmExe<BabyBear>) -> ChipMapping {
        let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
        ChipMapping {
            pc_to_chip: exe
                .program
                .instructions_and_debug_infos
                .iter()
                .map(|slot| {
                    if slot
                        .as_ref()
                        .is_some_and(|(insn, _)| insn.opcode == terminate_opcode)
                    {
                        TraceChipIndex::NoChip
                    } else {
                        TraceChipIndex::Chip(AirIndex::new(TEST_CHIP))
                    }
                })
                .collect(),
            chip_widths: None,
        }
    }

    fn phantom_chip_mapping(exe: &VmExe<BabyBear>) -> ChipMapping {
        let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
        let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
        ChipMapping {
            pc_to_chip: exe
                .program
                .instructions_and_debug_infos
                .iter()
                .map(|slot| match slot {
                    Some((insn, _)) if insn.opcode == terminate_opcode => TraceChipIndex::NoChip,
                    Some((insn, _)) if insn.opcode == phantom_opcode => {
                        TraceChipIndex::Chip(AirIndex::new(PHANTOM_CHIP))
                    }
                    Some(_) => TraceChipIndex::Chip(AirIndex::new(TEST_CHIP)),
                    None => TraceChipIndex::NoChip,
                })
                .collect(),
            chip_widths: None,
        }
    }

    #[test]
    fn classifier_flags_rv64im_only_vs_extension_using_exes() {
        let base = rv64im_memory_exe();
        assert_eq!(
            classify_preflight_opcodes(&base),
            RvrPreflightOpcodeClass::Supported
        );
        assert!(classify_preflight_opcodes(&base).is_supported());

        let extension = VmExe::new(Program::from_instructions(&[reveal_like_store()]));
        assert_eq!(
            classify_preflight_opcodes(&extension),
            RvrPreflightOpcodeClass::Unsupported {
                pc: 0,
                opcode: VmOpcode::from_usize(OPCODE_STORED),
            }
        );

        let non_memory_store = VmExe::new(Program::from_instructions(&[deferral_like_store()]));
        assert_eq!(
            classify_preflight_opcodes(&non_memory_store),
            RvrPreflightOpcodeClass::Unsupported {
                pc: 0,
                opcode: VmOpcode::from_usize(OPCODE_STORED),
            }
        );
    }

    #[test]
    fn preflight_compiles_and_logs_rv64im_program() {
        // This test asserts the verbose-log tracer contract (the path
        // non-migrated opcodes take), so opt out of R3 inline records — the
        // default compile suppresses AddSub memory-log entries. Safe under
        // nextest (one process per test).
        std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
        let exe = rv64im_memory_exe();
        let chips = chip_mapping(&exe);
        let compiled = compile_preflight(&exe, &chips, None).expect("compile preflight");
        assert!(compiled.artifact_dir().is_some());

        let mut vm_state: VmState = VmState::initial(
            &test_system_config(),
            &exe.init_memory,
            exe.pc_start,
            Streams::default(),
        );
        let mut program_log = vec![ProgramLogEntry::default(); 64];
        let mut memory_log = vec![MemoryLogEntry::default(); 64];
        let mut chip_counts = vec![0u32; 4];
        let mut tracer =
            PreflightTracerData::new(&mut program_log, &mut memory_log, &mut chip_counts);
        let mut shadows = TestShadows::new(&vm_state, 64);
        shadows.attach(&mut vm_state, &mut tracer);

        let state = execute_preflight_for_test(&compiled, &mut vm_state, &mut tracer)
            .expect("execute preflight");

        let program_len = tracer.program_log_len as usize;
        let memory_len = tracer.memory_log_len as usize;
        assert_eq!(program_len, state.mode_state.instret as usize);
        assert!(program_len > 0);
        assert!(memory_len > 0);
        assert_eq!(chip_counts[TEST_CHIP as usize], 15);

        let valid_pcs = exe
            .program
            .enumerate_by_pc()
            .into_iter()
            .map(|(pc, _, _)| pc)
            .collect::<HashSet<_>>();
        for entry in &program_log[..program_len] {
            assert!(
                valid_pcs.contains(&entry.pc()),
                "invalid pc {:#x}",
                entry.pc()
            );
        }
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(ProgramLogEntry::pc)
                .collect::<Vec<_>>(),
            (0..program_len as u32)
                .map(|idx| idx * DEFAULT_PC_STEP)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 3, 5, 9, 13, 15, 18, 21, 25, 29, 33, 37, 40, 44, 48, 52]
        );
        assert!(program_log[..program_len]
            .windows(2)
            .all(|pair| pair[0].timestamp <= pair[1].timestamp));
        assert!(memory_log[..memory_len]
            .windows(2)
            .all(|pair| pair[0].timestamp < pair[1].timestamp));
        assert_eq!(memory_len, 41);
        assert_eq!(tracer.timestamp, 52);
        assert!(memory_log[..memory_len]
            .iter()
            .filter(|entry| entry.addr_space == RV64_MEMORY_AS as u8)
            .all(|entry| entry.address == 64 && entry.width == 8));

        assert!(memory_log[..memory_len].iter().all(|entry| entry.width > 0
            && matches!(
                entry.kind,
                PREFLIGHT_MEMORY_KIND_READ | PREFLIGHT_MEMORY_KIND_WRITE
            )));
        assert!(memory_log[..memory_len].iter().any(|entry| entry.addr_space
            == RV64_MEMORY_AS as u8
            && entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
            && entry.address == 64
            && entry.value == 7));
        assert!(memory_log[..memory_len].iter().any(|entry| entry.addr_space
            == RV64_MEMORY_AS as u8
            && entry.kind == PREFLIGHT_MEMORY_KIND_READ
            && entry.address == 64
            && entry.value == 7));
        assert_eq!(
            memory_log[..memory_len]
                .iter()
                .filter(|entry| entry.addr_space == RV64_MEMORY_AS as u8
                    && entry.kind == PREFLIGHT_MEMORY_KIND_READ
                    && entry.address == 64
                    && entry.value == 7)
                .count(),
            1
        );
        assert!(memory_log[..memory_len]
            .iter()
            .any(|entry| entry.addr_space == RV64_MEMORY_AS as u8
                && entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.value == 0x5a07));
        assert!(memory_log[..memory_len]
            .iter()
            .any(|entry| entry.addr_space == RV64_MEMORY_AS as u8
                && entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.value == 0x005a5a07));
        assert!(memory_log[..memory_len]
            .iter()
            .any(|entry| entry.addr_space == RV64_MEMORY_AS as u8
                && entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.value == 0x0000005a005a5a07));
        assert!(memory_log[..memory_len]
            .iter()
            .any(|entry| entry.addr_space == RV64_MEMORY_AS as u8
                && entry.kind == PREFLIGHT_MEMORY_KIND_READ
                && entry.value == 0x0000005a005a5a07));

        let register_entries = memory_log[..memory_len]
            .iter()
            .filter(|entry| entry.addr_space == RV64_REGISTER_AS as u8)
            .collect::<Vec<_>>();
        assert!(
            !register_entries.is_empty(),
            "preflight must log AS_REGISTER accesses"
        );
        assert!(register_entries.iter().all(|entry| entry.width == 8));
        assert!(register_entries
            .iter()
            .any(|entry| entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.address == reg(1) as u64
                && entry.value == 64));
        assert!(register_entries
            .iter()
            .any(|entry| entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.address == reg(2) as u64
                && entry.value == 7));
        assert!(register_entries
            .iter()
            .any(|entry| entry.kind == PREFLIGHT_MEMORY_KIND_READ
                && entry.address == reg(2) as u64
                && entry.value == 7));
        assert!(register_entries
            .iter()
            .any(|entry| entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.address == reg(3) as u64
                && entry.value == 7));
        assert!(!register_entries
            .iter()
            .any(|entry| entry.kind == PREFLIGHT_MEMORY_KIND_WRITE && entry.address == 0));
    }

    #[test]
    fn preflight_phantoms_tick_shared_timestamp_and_chip_counts() {
        // Verbose-log contract test; opt out of R3 inline records (see
        // `preflight_compiles_and_logs_rv64im_program`).
        std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
        let exe = phantom_timestamp_exe();
        let chips = phantom_chip_mapping(&exe);
        let compiled = compile_preflight(&exe, &chips, None).expect("compile preflight");

        let mut vm_state: VmState = VmState::initial(
            &test_system_config(),
            &exe.init_memory,
            exe.pc_start,
            Streams::default(),
        );
        let mut program_log = vec![ProgramLogEntry::default(); 32];
        let mut memory_log = vec![MemoryLogEntry::default(); 32];
        let mut chip_counts = vec![0u32; 4];
        let mut tracer =
            PreflightTracerData::new(&mut program_log, &mut memory_log, &mut chip_counts);
        let mut shadows = TestShadows::new(&vm_state, 32);
        shadows.attach(&mut vm_state, &mut tracer);

        let state = execute_preflight_for_test(&compiled, &mut vm_state, &mut tracer)
            .expect("execute preflight");

        let program_len = tracer.program_log_len as usize;
        let memory_len = tracer.memory_log_len as usize;
        assert_eq!(program_len, state.mode_state.instret as usize);
        assert_eq!(program_len, 8);
        assert_eq!(memory_len, 10);
        assert_eq!(chip_counts[TEST_CHIP as usize], 4);
        assert_eq!(chip_counts[PHANTOM_CHIP as usize], 3);

        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(ProgramLogEntry::pc)
                .collect::<Vec<_>>(),
            (0..program_len as u32)
                .map(|idx| idx * DEFAULT_PC_STEP)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 3, 4, 6, 7, 11, 12, 16]
        );
        assert_eq!(
            memory_log[..memory_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5, 7, 8, 9, 12, 13, 15]
        );
        assert_eq!(tracer.timestamp, 16);

        let data_memory_timestamps = memory_log[..memory_len]
            .iter()
            .filter(|entry| entry.addr_space == RV64_MEMORY_AS as u8)
            .map(|entry| entry.timestamp)
            .collect::<Vec<_>>();
        assert_eq!(data_memory_timestamps, vec![9, 13]);

        let phantom_timestamps = [3, 6, 11];
        assert!(phantom_timestamps
            .iter()
            .all(|timestamp| !memory_log[..memory_len]
                .iter()
                .any(|entry| entry.timestamp == *timestamp)));
    }
}

/// R4: how a record-arena flavor stages itself as a C arena-native write
/// target and adopts the written records afterwards. The staged value must
/// not reallocate between staging and finish (the ChipRecordBuf holds a raw
/// pointer into its heap buffer; heap pointers are stable under moves).
pub trait RvrArenaNativeTarget: crate::arch::Arena + Sized {
    /// Allocate the arena/backing for `height` records and return the
    /// descriptor aiming the generated C at it.
    fn stage_arena_native(
        height: usize,
        width: usize,
        geometry: &super::ArenaNativeGeometry,
    ) -> (Self, ChipRecordBuf);

    /// Stage from the per-executor arena-native backing pool. Arena flavors keep the ordinary
    /// allocation path for direct callers; the proving loop supplies the AIR identity and pool.
    fn stage_arena_native_pooled(
        height: usize,
        width: usize,
        geometry: &super::ArenaNativeGeometry,
        _air: usize,
        _pool: &RvrPreflightBufferPool,
    ) -> (Self, ChipRecordBuf) {
        Self::stage_arena_native(height, width, geometry)
    }

    /// Commit `written_records` C-written records (cursor/offset bookkeeping
    /// only — the bytes are already in place).
    fn finish_arena_native(
        &mut self,
        written_records: usize,
        geometry: &super::ArenaNativeGeometry,
    );

    /// Commit a target when the generated C also reports its exact byte
    /// cursor. Fixed-shape implementations inherit the count-based behavior;
    /// packed variable-row arenas override it.
    fn finish_arena_native_sized(
        &mut self,
        written_records: usize,
        written_bytes: usize,
        geometry: &super::ArenaNativeGeometry,
    ) {
        assert_eq!(
            written_bytes,
            written_records * geometry.stride_dense(),
            "fixed-shape arena-native byte cursor mismatch"
        );
        self.finish_arena_native(written_records, geometry);
    }

    /// Attach the committed G2 segment identity to an opaque-final arena for
    /// per-segment CUDA H2D timing. Non-dense arenas do not need the marker.
    fn set_rvr_g2_segment_id(&mut self, _segment_id: u32) {}

    /// G2: stage this arena as a compact WIRE record target — the C writes
    /// packed wire records (stride = `wire_size`) directly into the arena's
    /// aligned backing, so adoption is cursor bookkeeping instead of an
    /// alloc + memcpy. Only a dense byte arena can hold wire records (its
    /// consumer decodes them, e.g. the GPU compact kernels); flavors whose
    /// consumers expect expanded records must never be asked.
    fn stage_rvr_wire(records_cap: usize, wire_size: usize) -> (Self, ChipRecordBuf);

    /// Stage from the pre-touched per-executor backing pool. Arena flavors
    /// without recyclable byte backings retain the ordinary implementation.
    fn stage_rvr_wire_pooled(
        records_cap: usize,
        wire_size: usize,
        _air: usize,
        _pool: &RvrPreflightBufferPool,
    ) -> (Self, ChipRecordBuf) {
        Self::stage_rvr_wire(records_cap, wire_size)
    }

    /// Commit `written_records` C-written wire records and mark the arena as
    /// wire-mode for its consumer.
    fn finish_rvr_wire(&mut self, written_records: usize, wire_size: usize);
}

fn arena_native_flags(geometry: &super::ArenaNativeGeometry) -> u32 {
    let mut flags = PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL;
    if matches!(
        geometry.layout,
        super::ArenaNativeLayout::Custom {
            residual_memory_chronology: true,
            ..
        } | super::ArenaNativeLayout::CustomVariableRows {
            residual_memory_chronology: true
        }
    ) {
        flags |= PREFLIGHT_CHIP_RECORD_FLAG_RESIDUAL_MEMORY_CHRONOLOGY;
    }
    if matches!(
        geometry.layout,
        super::ArenaNativeLayout::CustomVariableRows { .. }
    ) {
        flags |= PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROWS;
    }
    flags
}

impl<F: openvm_stark_backend::p3_field::Field> RvrArenaNativeTarget
    for crate::arch::MatrixRecordArena<F>
{
    fn stage_arena_native(
        height: usize,
        width: usize,
        geometry: &super::ArenaNativeGeometry,
    ) -> (Self, ChipRecordBuf) {
        use crate::arch::Arena;
        let mut arena = Self::with_capacity(height, width);
        let stride = (arena.width * std::mem::size_of::<F>()) as u32;
        let cap_bytes = (arena.trace_buffer.len() * std::mem::size_of::<F>()) as u32;
        let buf = ChipRecordBuf {
            base: arena.trace_buffer.as_mut_ptr().cast(),
            len: 0,
            cap: cap_bytes - cap_bytes % stride,
            stride,
            core_off: if matches!(
                geometry.layout,
                super::ArenaNativeLayout::CustomVariableRows { .. }
            ) {
                0
            } else {
                geometry.core_off_matrix as u32
            },
            flags: arena_native_flags(geometry)
                | if matches!(
                    geometry.layout,
                    super::ArenaNativeLayout::CustomVariableRows { .. }
                ) {
                    PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE
                } else {
                    0
                },
        };
        (arena, buf)
    }

    fn stage_arena_native_pooled(
        height: usize,
        width: usize,
        geometry: &super::ArenaNativeGeometry,
        air: usize,
        pool: &RvrPreflightBufferPool,
    ) -> (Self, ChipRecordBuf) {
        let padded_height = next_power_of_two_or_zero(height);
        let stride_bytes = width * std::mem::size_of::<F>();
        let capacity_bytes = padded_height * stride_bytes;
        let key =
            super::preflight_pool::ArenaNativeBackingKey::new(air, stride_bytes, capacity_bytes);
        let mut arena = Self::with_recycled_rvr_capacity(height, width, key, pool.clone());
        let stride = stride_bytes as u32;
        let cap_bytes = capacity_bytes as u32;
        let buf = ChipRecordBuf {
            base: arena.trace_buffer.as_mut_ptr().cast(),
            len: 0,
            cap: cap_bytes,
            stride,
            core_off: if matches!(
                geometry.layout,
                super::ArenaNativeLayout::CustomVariableRows { .. }
            ) {
                0
            } else {
                geometry.core_off_matrix as u32
            },
            flags: arena_native_flags(geometry)
                | if matches!(
                    geometry.layout,
                    super::ArenaNativeLayout::CustomVariableRows { .. }
                ) {
                    PREFLIGHT_CHIP_RECORD_FLAG_VARIABLE_ROW_STRIDE
                } else {
                    0
                },
        };
        (arena, buf)
    }

    fn finish_arena_native(&mut self, written_records: usize, _: &super::ArenaNativeGeometry) {
        self.trace_offset = written_records * self.width;
    }

    fn finish_arena_native_sized(
        &mut self,
        written_records: usize,
        written_bytes: usize,
        geometry: &super::ArenaNativeGeometry,
    ) {
        if matches!(
            geometry.layout,
            super::ArenaNativeLayout::CustomVariableRows { .. }
        ) {
            assert_eq!(
                written_bytes,
                written_records * self.width * std::mem::size_of::<F>(),
                "variable-row Matrix arena byte cursor mismatch"
            );
            self.trace_offset = written_records * self.width;
        } else {
            let _ = written_bytes;
            self.finish_arena_native(written_records, geometry);
        }
    }

    fn stage_rvr_wire(_records_cap: usize, _wire_size: usize) -> (Self, ChipRecordBuf) {
        unreachable!(
            "compact wire staging requires a dense record arena; the Matrix flavor's \
             consumers expect expanded rows"
        )
    }

    fn finish_rvr_wire(&mut self, _written_records: usize, _wire_size: usize) {
        unreachable!("compact wire staging requires a dense record arena")
    }
}

impl RvrArenaNativeTarget for crate::arch::DenseRecordArena {
    fn stage_arena_native(
        height: usize,
        _width: usize,
        geometry: &super::ArenaNativeGeometry,
    ) -> (Self, ChipRecordBuf) {
        let stride = geometry.stride_dense();
        let mut arena = Self::with_byte_capacity(height * stride);
        // with_byte_capacity positions the cursor at the aligned start; the C
        // writes from exactly there.
        let offset = arena.records_buffer.position() as usize;
        debug_assert_eq!(
            (arena.records_buffer.get_ref().as_ptr() as usize + offset) % 32,
            0,
            "dense arena-native base must be 32-aligned"
        );
        let base = unsafe { arena.records_buffer.get_mut().as_mut_ptr().add(offset) };
        let buf = ChipRecordBuf {
            base,
            len: 0,
            cap: (height * stride) as u32,
            stride: stride as u32,
            core_off: if matches!(
                geometry.layout,
                super::ArenaNativeLayout::CustomVariableRows { .. }
            ) {
                0
            } else {
                geometry.core_off_dense() as u32
            },
            flags: arena_native_flags(geometry),
        };
        (arena, buf)
    }

    fn stage_arena_native_pooled(
        height: usize,
        _width: usize,
        geometry: &super::ArenaNativeGeometry,
        air: usize,
        pool: &RvrPreflightBufferPool,
    ) -> (Self, ChipRecordBuf) {
        let stride = geometry.stride_dense();
        let capacity_bytes = height * stride;
        let key = super::preflight_pool::ArenaNativeBackingKey::new(air, stride, capacity_bytes);
        let mut arena =
            Self::with_recycled_rvr_arena_native_capacity(capacity_bytes, key, pool.clone());
        let offset = arena.records_buffer.position() as usize;
        debug_assert_eq!(
            (arena.records_buffer.get_ref().as_ptr() as usize + offset) % 32,
            0,
            "recycled dense arena-native base must be 32-aligned"
        );
        let base = unsafe { arena.records_buffer.get_mut().as_mut_ptr().add(offset) };
        let buf = ChipRecordBuf {
            base,
            len: 0,
            cap: capacity_bytes as u32,
            stride: stride as u32,
            core_off: if matches!(
                geometry.layout,
                super::ArenaNativeLayout::CustomVariableRows { .. }
            ) {
                0
            } else {
                geometry.core_off_dense() as u32
            },
            flags: arena_native_flags(geometry),
        };
        (arena, buf)
    }

    fn finish_arena_native(
        &mut self,
        written_records: usize,
        geometry: &super::ArenaNativeGeometry,
    ) {
        let offset = {
            let ptr = self.records_buffer.get_ref().as_ptr() as usize;
            (32 - ptr % 32) % 32
        };
        self.records_buffer
            .set_position((offset + written_records * geometry.stride_dense()) as u64);
    }

    fn finish_arena_native_sized(
        &mut self,
        written_records: usize,
        written_bytes: usize,
        geometry: &super::ArenaNativeGeometry,
    ) {
        if matches!(
            geometry.layout,
            super::ArenaNativeLayout::CustomVariableRows { .. }
        ) {
            let offset = {
                let ptr = self.records_buffer.get_ref().as_ptr() as usize;
                (32 - ptr % 32) % 32
            };
            assert!(
                offset + written_bytes <= self.records_buffer.get_ref().len(),
                "variable-row arena-native byte cursor exceeds backing"
            );
            self.records_buffer
                .set_position((offset + written_bytes) as u64);
            self.rvr_variable_rows = Some(written_records);
        } else {
            assert_eq!(
                written_bytes,
                written_records * geometry.stride_dense(),
                "fixed-shape arena-native byte cursor mismatch"
            );
            self.finish_arena_native(written_records, geometry);
        }
    }

    fn set_rvr_g2_segment_id(&mut self, segment_id: u32) {
        self.rvr_g2_segment_id = Some(segment_id);
    }

    fn stage_rvr_wire(records_cap: usize, wire_size: usize) -> (Self, ChipRecordBuf) {
        let mut arena = Self::with_byte_capacity(records_cap * wire_size);
        let offset = arena.records_buffer.position() as usize;
        debug_assert_eq!(
            (arena.records_buffer.get_ref().as_ptr() as usize + offset) % 32,
            0,
            "dense wire base must be 32-aligned"
        );
        let base = unsafe { arena.records_buffer.get_mut().as_mut_ptr().add(offset) };
        let buf = ChipRecordBuf {
            base,
            len: 0,
            cap: (records_cap * wire_size) as u32,
            // Compact wire: the stride IS the packed record size (the same
            // descriptor shape the pooled record buffers use).
            stride: wire_size as u32,
            core_off: 0,
            flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
        };
        (arena, buf)
    }

    fn stage_rvr_wire_pooled(
        records_cap: usize,
        wire_size: usize,
        air: usize,
        pool: &RvrPreflightBufferPool,
    ) -> (Self, ChipRecordBuf) {
        let len = records_cap * wire_size;
        let Some(backing) = pool.take_wire_backing(air, len) else {
            return Self::stage_rvr_wire(records_cap, wire_size);
        };
        let mut arena = Self::from_recycled_wire_backing(backing, air, pool.clone());
        let offset = arena.records_buffer.position() as usize;
        debug_assert_eq!(
            (arena.records_buffer.get_ref().as_ptr() as usize + offset) % 32,
            0,
            "recycled dense wire base must be 32-aligned"
        );
        let base = unsafe { arena.records_buffer.get_mut().as_mut_ptr().add(offset) };
        let buf = ChipRecordBuf {
            base,
            len: 0,
            cap: len as u32,
            stride: wire_size as u32,
            core_off: 0,
            flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
        };
        (arena, buf)
    }

    fn finish_rvr_wire(&mut self, written_records: usize, wire_size: usize) {
        let offset = {
            let ptr = self.records_buffer.get_ref().as_ptr() as usize;
            (32 - ptr % 32) % 32
        };
        self.records_buffer
            .set_position((offset + written_records * wire_size) as u64);
        // The mode travels with the data: this routes the arena to the chips'
        // compact-decode branch instead of the expanded-record kernels.
        self.rvr_wire = true;
    }
}
