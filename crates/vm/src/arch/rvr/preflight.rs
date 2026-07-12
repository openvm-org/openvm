//! Preflight tracer ABI mirror for rvr-generated native execution.

use std::{collections::BTreeMap, mem::MaybeUninit, sync::Arc};

use openvm_instructions::{
    exe::VmExe,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rvr_openvm_lift::{RvrRuntimeExtension, TraceChipIndex};

use super::{
    bridge::{map_rvr_execute_error, public_values_slice},
    compile::{ChipMapping, RvrCompiled},
    execute::execute_preflight as execute_preflight_raw,
    preflight_normalizer::{
        build_preflight_replay, PreflightMemoryAccessAux, PreflightShadowsView,
    },
    preflight_pool::{scrub_shadows, RvrPreflightBufferPool},
    state::{TracerPayload, TracerPtr},
};
use crate::{
    arch::{
        interpreter_preflight::PreflightInterpretedInstance, ExecutionError, ExecutionState,
        Streams, SystemConfig, VmState, BLOCK_FE_WIDTH,
    },
    system::{
        memory::{merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory},
        SystemRecords,
    },
};

pub const PREFLIGHT_TRACER_KIND: u32 = rvr_openvm_ext_ffi_common::PREFLIGHT_TRACER_KIND;
pub const PREFLIGHT_INITIAL_TIMESTAMP: u32 = rvr_openvm_ext_ffi_common::PREFLIGHT_INITIAL_TIMESTAMP;
pub const PREFLIGHT_MEMORY_KIND_READ: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_READ;
pub const PREFLIGHT_MEMORY_KIND_WRITE: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_WRITE;
pub const PREFLIGHT_MEMORY_KIND_TOUCH: u8 = rvr_openvm_ext_ffi_common::PREFLIGHT_MEMORY_KIND_TOUCH;
pub const PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL;
pub const PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW: u32 =
    rvr_openvm_ext_ffi_common::PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW;
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

/// C-compatible preflight program log entry.
///
/// Layout matches `ProgramLogEntry` in `openvm_tracer_preflight.h`.
/// `opcode` is reserved for a future richer emitted hook; M1 logs use `pc`
/// and recover opcode metadata from the `VmExe`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramLogEntry {
    pub opcode: u16,
    pub _pad0: u16,
    pub timestamp: u32,
    pub pc: u64,
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

/// C-compatible preflight touched-block entry.
///
/// Layout matches `TouchedBlock` in `openvm_tracer_preflight.h`. Records a block
/// touched for the first time this segment; the host finalizes `touched_memory`
/// from this list (final value from live memory, final timestamp from the
/// shadow) in O(touched-blocks).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TouchedBlock {
    pub addr_space: u32,
    pub block_addr: u32,
}

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

    pub fn set_delta_records(&mut self, delta_records: &mut ChipRecordBuf) {
        self.delta_records = delta_records;
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
        }
    }
}

impl TracerPayload for PreflightTracerData {
    const KIND: u32 = PREFLIGHT_TRACER_KIND;
}

pub type PreflightTracer = TracerPtr<PreflightTracerData>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightRawLogs {
    pub program_log: Vec<ProgramLogEntry>,
    pub memory_log: Vec<MemoryLogEntry>,
    pub chip_counts: Vec<u32>,
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
    pool: RvrPreflightBufferPool,
}

impl RvrDeltaRecords {
    fn new(backing: Vec<u8>, offset: usize, capacity: usize, pool: RvrPreflightBufferPool) -> Self {
        Self {
            backing: Some(backing),
            offset,
            written: 0,
            // Until native execution returns, conservatively assume C may
            // have dirtied the complete target on an error path.
            dirty_len: offset + capacity,
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
    /// R3: per program slot, whether that instruction emits an inline compact
    /// record — its memory-log entries are suppressed, so record assembly must
    /// skip the log assembler for it and consume `inline_records` instead.
    pub inline_pc_slots: Arc<Vec<bool>>,
    /// R4: `(air_idx, written_record_count)` for airs whose records the C
    /// wrote arena-native into caller-provided targets. Record assembly must
    /// skip BOTH the log assembler and the inline assembler for these airs
    /// and only verify the count against the program log.
    pub arena_native_written: Vec<(usize, u32)>,
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
) -> Result<RvrPreflightOutput<F>, ExecutionError>
where
    F: PrimeField32,
{
    assert!(
        arena_targets.is_none() || (num_insns.is_some() && record_capacity_rows.is_some()),
        "arena-native targets require the single-shot proving path"
    );
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
        // The log buffers are write-only from the C tracer and prefix-read by
        // the host, so they are allocated uninitialized: the zero-fill of
        // multi-GB capacities was a real per-call cost on large segments.
        // They come from (and return to) the cross-segment pool: re-faulting
        // hundreds of MB of fresh mappings per segment was the other measured
        // per-call cost.
        let mut program_log = pool.take_program_log(program_log_cap);
        let mut memory_log = pool.take_memory_log(memory_log_cap);
        let mut chip_counts = vec![0u32; chip_counts_len.max(1)];
        let mut exec_frequencies = vec![0u32; exe.program.num_defined_instructions()];
        // Shadows are all-zero at segment start (0 = untouched this segment);
        // pooled reuse preserves that via the exact touched-list scrub below.
        // Distinct touched blocks are bounded by the number of memory-log
        // entries, so sizing `touched` to `memory_log_cap` guarantees it
        // never overflows.
        let (mut shadow_register, mut shadow_memory, mut shadow_public_values) =
            pool.take_shadows(reg_shadow_blocks, mem_shadow_blocks, pv_shadow_blocks);
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
        let mut touched = pool.take_touched(memory_log_cap + inline_touch_slack);
        let pv_base = public_values_slice(&mut run_state.memory.memory).as_mut_ptr();

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
                if inline_meta.delta_records {
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
        let delta_capacity = program_log_cap.saturating_mul(PREFLIGHT_DELTA_RECORD_SIZE);
        let mut delta_output = if inline_meta.delta_records {
            let backing = pool.take_delta_backing(delta_capacity);
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
            // The CUDA path sources this stream from the page-locked record
            // pool. Mirror its ready-buffer contract on CPU: fault one byte
            // per page before the native execute+emit clock starts, so lazy
            // zero-page faults are never charged as record emission.
            for page in (0..delta_capacity).step_by(4096) {
                unsafe { std::ptr::write_volatile(aligned_base.add(page), 0) };
            }
            if delta_capacity != 0 {
                unsafe { std::ptr::write_volatile(aligned_base.add(delta_capacity - 1), 0) };
            }
            delta_record = ChipRecordBuf {
                base: aligned_base.cast(),
                len: 0,
                cap: u32::try_from(delta_capacity).expect("delta capacity exceeds u32"),
                stride: PREFLIGHT_DELTA_RECORD_SIZE as u32,
                core_off: 0,
                flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
            };
        }
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

        let mut tracer =
            PreflightTracerData::new_uninit(&mut program_log, &mut memory_log, &mut chip_counts);
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
        if inline_meta.delta_records {
            tracer.set_delta_records(&mut delta_record);
        }

        let native_execute_started = std::time::Instant::now();
        let run_result = execute_preflight_raw(
            compiled,
            runtime_hooks,
            &mut run_state,
            &mut tracer,
            num_insns,
        );
        let native_execute_elapsed = native_execute_started.elapsed();
        // Drain non-temporal record stores before either consuming the
        // buffers or propagating an execution error. On error, propagation
        // drops the RAII delta lease and may queue its pinned backing for
        // asynchronous cleaning/reuse.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_sfence();
        }
        let run_result = run_result.map_err(map_rvr_execute_error)?;
        if let Some(delta) = delta_output.as_mut() {
            delta.set_written(delta_record.len as usize);
        }
        if std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "OPENVM_RVR_NATIVE_EXEC_EMIT_US={} delta={} delta_records={} delta_bytes={}",
                native_execute_elapsed.as_micros(),
                inline_meta.delta_records as u8,
                delta_record.len / PREFLIGHT_DELTA_RECORD_SIZE as u32,
                delta_record.len,
            );
        }
        if let Some(target_instret) = num_insns {
            if run_result.suspended && run_result.state.instret != target_instret {
                return Err(ExecutionError::RvrExecution(format!(
                    "mid-block rvr preflight suspension unsupported: requested num_insns={target_instret}, retired instret={} at an rvr basic-block boundary",
                    run_result.state.instret
                )));
            }
        }
        if delta_record.flags & PREFLIGHT_CHIP_RECORD_FLAG_OVERFLOW != 0 {
            return Err(ExecutionError::RvrExecution(
                "stage-2 delta record reservation overflow or ABI mismatch".to_string(),
            ));
        }

        let program_len = tracer.program_log_len as usize;
        let memory_len = tracer.memory_log_len as usize;
        let touched_len = tracer.touched_len as usize;
        // With inline records (R3), migrated opcodes touch without logging, so
        // `touched` can outgrow the memory log; growing `memory_log_cap` grows
        // the touched buffer with it.
        if program_len > program_log_cap
            || memory_len > memory_log_cap
            || touched_len > touched.len()
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
                     touched {touched_len}/{}, delta {}/{} — capacity-model bug",
                    touched.len(),
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
            pool.recycle_raw_uninit(program_log, memory_log, touched);
            for buf in record_bufs {
                pool.recycle_record_buf(buf);
            }
            continue;
        }

        // SAFETY: the C tracer fully wrote the first `*_len` entries of each
        // log (its append helpers bounds-check against the caps, and the
        // overflow check above rejected any run whose counters passed them).
        let program_log = unsafe { assume_init_prefix(program_log, program_len) };
        let memory_log = unsafe { assume_init_prefix(memory_log, memory_len) };
        let touched = unsafe { assume_init_prefix(touched, touched_len) };

        let shadows = PreflightShadowsView {
            register: &shadow_register,
            memory: &shadow_memory,
            public_values: &shadow_public_values,
        };
        let replay =
            build_preflight_replay::<F>(&run_state.memory, &shadows, &touched, &memory_log)
                .map_err(|err| ExecutionError::RvrExecution(err.to_string()))?;
        run_state
            .memory
            .memory
            .extend_touched_pages_from_touched(&replay.touched_memory);
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
        pool.recycle_shadows(shadow_register, shadow_memory, shadow_public_values);
        pool.recycle_touched(touched);
        let filtered_exec_frequencies = exec_frequencies;
        let to_state = ExecutionState::new(run_state.pc(), tracer.timestamp);
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code: run_result.exit_code,
            filtered_exec_frequencies,
            touched_memory: replay.touched_memory,
        };

        // R3: harvest each migrated chip's inline records (the C-advanced
        // cursor gives the written length).
        let arena_native_written: Vec<(usize, u32)> = inline_meta
            .airs
            .iter()
            .filter_map(|&(air_idx, _)| {
                arena_targets
                    .is_some_and(|targets| targets.contains_key(&air_idx))
                    .then(|| {
                        let buf = &chip_records[air_idx];
                        assert_eq!(
                            buf.len % buf.stride,
                            0,
                            "arena-native cursor for air {air_idx} is not a whole record count"
                        );
                        (air_idx, buf.len / buf.stride)
                    })
            })
            .collect();
        let inline_records: Vec<RvrInlineChipRecords> = if inline_meta.delta_records {
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

        return Ok(RvrPreflightOutput {
            system_records,
            raw_logs: PreflightRawLogs {
                program_log,
                memory_log,
                chip_counts,
            },
            access_aux: replay.access_aux,
            to_state: run_state,
            instret: run_result.state.instret,
            suspended: run_result.suspended,
            inline_records,
            delta_records,
            inline_pc_slots: inline_meta.pc_slots.clone(),
            arena_native_written,
        });
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
mod tests {
    use std::collections::HashSet;

    use openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, SysPhantom, SystemOpcode, VmOpcode, DEFERRAL_AS,
    };
    use p3_baby_bear::BabyBear;
    use rvr_openvm_ext_ffi_common::AS_PUBLIC_VALUES;
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
        utils::test_system_config,
    };

    /// Owns the per-address-space timestamp shadows + touched buffer for a
    /// direct-`execute_preflight_for_test` call and attaches them to the tracer.
    struct TestShadows {
        register: Vec<u32>,
        memory: Vec<u32>,
        public_values: Vec<u32>,
        touched: Vec<TouchedBlock>,
    }

    impl TestShadows {
        fn new(vm_state: &VmState<BabyBear>, touched_cap: usize) -> Self {
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

        fn attach(&mut self, vm_state: &mut VmState<BabyBear>, tracer: &mut PreflightTracerData) {
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

    const OPCODE_ADD: usize = 0x200;
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
            VmOpcode::from_usize(OPCODE_ADD),
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

        let mut vm_state: VmState<BabyBear> = VmState::initial(
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
        assert_eq!(program_len, state.instret as usize);
        assert!(program_len > 0);
        assert!(memory_len > 0);
        assert_eq!(chip_counts[TEST_CHIP as usize], 15);

        let valid_pcs = exe
            .program
            .enumerate_by_pc()
            .into_iter()
            .map(|(pc, _, _)| u64::from(pc))
            .collect::<HashSet<_>>();
        for entry in &program_log[..program_len] {
            assert!(valid_pcs.contains(&entry.pc), "invalid pc {:#x}", entry.pc);
        }
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.pc)
                .collect::<Vec<_>>(),
            (0..program_len as u64)
                .map(|idx| idx * u64::from(DEFAULT_PC_STEP))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            (0..program_len as u32)
                .map(|idx| 1 + idx * 3)
                .collect::<Vec<_>>()
        );
        assert!(program_log[..program_len]
            .windows(2)
            .all(|pair| pair[0].timestamp <= pair[1].timestamp));
        assert!(memory_log[..memory_len]
            .windows(2)
            .all(|pair| pair[0].timestamp < pair[1].timestamp));
        assert_eq!(memory_len, 41);
        assert_eq!(tracer.timestamp, 46);
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

        let mut vm_state: VmState<BabyBear> = VmState::initial(
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
        assert_eq!(program_len, state.instret as usize);
        assert_eq!(program_len, 8);
        assert_eq!(memory_len, 10);
        assert_eq!(chip_counts[TEST_CHIP as usize], 4);
        assert_eq!(chip_counts[PHANTOM_CHIP as usize], 3);

        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.pc)
                .collect::<Vec<_>>(),
            (0..program_len as u32)
                .map(|idx| u64::from(idx * DEFAULT_PC_STEP))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            program_log[..program_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 4, 5, 8, 9, 12, 13, 16]
        );
        assert_eq!(
            memory_log[..memory_len]
                .iter()
                .map(|entry| entry.timestamp)
                .collect::<Vec<_>>(),
            vec![1, 3, 5, 7, 9, 10, 11, 13, 14, 15]
        );
        assert_eq!(tracer.timestamp, 16);

        let data_memory_timestamps = memory_log[..memory_len]
            .iter()
            .filter(|entry| entry.addr_space == RV64_MEMORY_AS as u8)
            .map(|entry| entry.timestamp)
            .collect::<Vec<_>>();
        assert_eq!(data_memory_timestamps, vec![11, 14]);

        let phantom_timestamps = [4, 8, 12];
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

    /// Commit `written_records` C-written records (cursor/offset bookkeeping
    /// only — the bytes are already in place).
    fn finish_arena_native(
        &mut self,
        written_records: usize,
        geometry: &super::ArenaNativeGeometry,
    );

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
            core_off: geometry.core_off_matrix as u32,
            flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
        };
        (arena, buf)
    }

    fn finish_arena_native(&mut self, written_records: usize, _: &super::ArenaNativeGeometry) {
        self.trace_offset = written_records * self.width;
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
            core_off: geometry.core_off_dense() as u32,
            flags: PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
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
