//! Preflight tracer ABI mirror for rvr-generated native execution.

use std::{collections::HashMap, sync::Arc};

use openvm_instructions::{
    exe::VmExe,
    program::DEFAULT_PC_STEP,
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
    state::{TracerPayload, TracerPtr},
};
use crate::{
    arch::{
        interpreter_preflight::PreflightInterpretedInstance, DenseRecordArena, ExecutionError,
        ExecutionState, InlineRecordBytes, Streams, SystemConfig, VmState, BLOCK_FE_WIDTH,
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
/// R3: byte size of one compact AddSub inline record (see the C
/// `PreflightAddSubRecord`). Used to size the per-chip inline-record buffers.
pub const PREFLIGHT_ADDSUB_RECORD_SIZE: usize =
    rvr_openvm_ext_ffi_common::PREFLIGHT_ADDSUB_RECORD_SIZE;

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

/// C-compatible per-chip inline-record buffer descriptor (R3).
///
/// Layout must match the C `ChipRecordBuf` in `openvm_tracer_preflight.h`.
/// `base` points at a host-allocated, metered-height-sized byte buffer of
/// compact records for one chip; `len` is the byte cursor the C advances; `cap`
/// the byte capacity. A null `base` means the chip is not migrated to inline
/// records (it uses the verbose memory log).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChipRecordBuf {
    pub base: *mut u8,
    pub len: u32,
    pub cap: u32,
}

impl Default for ChipRecordBuf {
    fn default() -> Self {
        Self {
            base: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }
}

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
        }
    }

    /// Attach the per-chip inline-record buffers (R3). The slice must have one
    /// entry per chip (`chip_counts_len`); migrated chips carry a non-null
    /// `base` pointing at a metered-height-sized record buffer, others null.
    pub fn set_chip_records(&mut self, chip_records: &mut [ChipRecordBuf]) {
        debug_assert_eq!(chip_records.len() as u32, self.chip_counts_len);
        self.chip_records = chip_records.as_mut_ptr();
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
/// generated C directly in the record arena's packed format (R3).
#[derive(Debug)]
pub struct RvrInlineChipRecords {
    pub air_idx: usize,
    /// Packed record stride in bytes (from the compile metadata).
    pub record_size: usize,
    pub records: InlineRecordBytes,
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
    /// R3: per program slot, whether that instruction emits an inline compact
    /// record — its memory-log entries are suppressed, so record assembly must
    /// skip the log assembler for it and consume `inline_records` instead.
    pub inline_pc_slots: Arc<Vec<bool>>,
}

pub struct RvrPreflightInstance<'a, F: PrimeField32> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    pub(crate) compiled: RvrCompiled,
    pub(crate) chip_counts_len: usize,
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
        }
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
            state,
            num_insns,
            None,
        )
    }
}

/// Executes a compiled preflight library. `record_capacity_rows`, when
/// present, bounds each migrated chip's inline-record count for this segment
/// (the metered per-AIR trace heights, indexed by AIR); without it the record
/// buffers fall back to the one-record-per-instruction bound of the program
/// log.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_rvr_preflight<F>(
    exe: &VmExe<F>,
    runtime_hooks: &[Box<dyn RvrRuntimeExtension>],
    compiled: &RvrCompiled,
    chip_counts_len: usize,
    state: VmState<GuestMemory>,
    num_insns: Option<u64>,
    record_capacity_rows: Option<&[u32]>,
) -> Result<RvrPreflightOutput<F>, ExecutionError>
where
    F: PrimeField32,
{
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
    let mut memory_log_cap = initial_memory_log_cap(program_log_cap);

    loop {
        let mut run_state = state.clone();
        let mut program_log = vec![ProgramLogEntry::default(); program_log_cap];
        let mut memory_log = vec![MemoryLogEntry::default(); memory_log_cap];
        let mut chip_counts = vec![0u32; chip_counts_len.max(1)];
        // Shadows are zeroed each run (0 = untouched this segment). Distinct
        // touched blocks are bounded by the number of memory-log entries, so
        // sizing `touched` to `memory_log_cap` guarantees it never overflows.
        let mut shadow_register = vec![0u32; reg_shadow_blocks];
        let mut shadow_memory = vec![0u32; mem_shadow_blocks];
        let mut shadow_public_values = vec![0u32; pv_shadow_blocks];
        let mut touched = vec![TouchedBlock::default(); memory_log_cap];
        let pv_base = public_values_slice(&mut run_state.memory.memory).as_mut_ptr();

        // R3: per migrated chip, an inline compact-record buffer in the record
        // arena's packed format (32-aligned start so the arena can adopt the
        // buffer zero-copy). Sized by the metered per-AIR trace heights when
        // available, else by one record per instruction.
        let inline_meta = compiled.inline_records();
        let mut record_bufs: Vec<(Vec<u8>, usize, usize)> = inline_meta
            .airs
            .iter()
            .map(|&(air, record_size)| {
                let rows = record_capacity_rows
                    .and_then(|heights| heights.get(air))
                    .map(|&height| height as usize)
                    .unwrap_or(program_log_cap);
                let cap_bytes = rows * record_size;
                let (buffer, offset) = DenseRecordArena::backing_with_capacity(cap_bytes);
                (buffer, offset, cap_bytes)
            })
            .collect();
        let mut chip_records = vec![ChipRecordBuf::default(); chip_counts_len.max(1)];
        for (&(air, _), (buffer, offset, cap_bytes)) in
            inline_meta.airs.iter().zip(record_bufs.iter_mut())
        {
            debug_assert!(air < chip_records.len(), "inline air {air} out of range");
            chip_records[air] = ChipRecordBuf {
                // SAFETY: `offset` is within the buffer (backing_with_capacity
                // over-allocates by the alignment slack).
                base: unsafe { buffer.as_mut_ptr().add(*offset) },
                len: 0,
                cap: *cap_bytes as u32,
            };
        }

        let mut tracer =
            PreflightTracerData::new(&mut program_log, &mut memory_log, &mut chip_counts);
        tracer.set_shadows(
            &mut shadow_register,
            &mut shadow_memory,
            &mut shadow_public_values,
            pv_base,
            &mut touched,
        );
        tracer.set_chip_records(&mut chip_records);

        let run_result = execute_preflight_raw(
            compiled,
            runtime_hooks,
            &mut run_state,
            &mut tracer,
            num_insns,
        )
        .map_err(map_rvr_execute_error)?;
        if let Some(target_instret) = num_insns {
            if run_result.suspended && run_result.state.instret != target_instret {
                return Err(ExecutionError::RvrExecution(format!(
                    "mid-block rvr preflight suspension unsupported: requested num_insns={target_instret}, retired instret={} at an rvr basic-block boundary",
                    run_result.state.instret
                )));
            }
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
        {
            program_log_cap = grow_capacity(program_log_cap, program_len);
            memory_log_cap = grow_capacity(memory_log_cap, memory_len.max(touched_len));
            continue;
        }

        program_log.truncate(program_len);
        memory_log.truncate(memory_len);

        let shadows = PreflightShadowsView {
            register: &shadow_register,
            memory: &shadow_memory,
            public_values: &shadow_public_values,
        };
        let replay = build_preflight_replay::<F>(
            &run_state.memory,
            &shadows,
            &touched[..touched_len],
            &memory_log,
        )
        .map_err(|err| ExecutionError::RvrExecution(err.to_string()))?;
        run_state
            .memory
            .memory
            .extend_touched_pages_from_touched(&replay.touched_memory);
        let filtered_exec_frequencies = filtered_exec_frequencies(exe, &program_log)?;
        let to_state = ExecutionState::new(run_state.pc(), tracer.timestamp);
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code: run_result.exit_code,
            filtered_exec_frequencies,
            touched_memory: replay.touched_memory,
        };

        // R3: harvest each migrated chip's inline records (the C-advanced
        // cursor gives the written length). The backing buffers move out
        // whole so a dense arena can adopt them zero-copy.
        let inline_records: Vec<RvrInlineChipRecords> = inline_meta
            .airs
            .iter()
            .zip(record_bufs.iter_mut())
            .map(|(&(air_idx, record_size), (buffer, _, _))| {
                let written = chip_records[air_idx].len as usize;
                RvrInlineChipRecords {
                    air_idx,
                    record_size,
                    records: InlineRecordBytes {
                        buffer: std::mem::take(buffer),
                        written,
                    },
                }
            })
            .collect();

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
            inline_pc_slots: inline_meta.pc_slots.clone(),
        });
    }
}

fn filtered_exec_frequencies<F: Field>(
    exe: &VmExe<F>,
    program_log: &[ProgramLogEntry],
) -> Result<Vec<u32>, ExecutionError> {
    let mut pc_to_filtered_idx = HashMap::new();
    let mut idx = 0usize;
    for (slot_idx, slot) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        if slot.is_some() {
            let pc = exe.program.pc_base + slot_idx as u32 * DEFAULT_PC_STEP;
            pc_to_filtered_idx.insert(u64::from(pc), idx);
            idx += 1;
        }
    }

    let mut frequencies = vec![0u32; idx];
    for entry in program_log {
        let &filtered_idx = pc_to_filtered_idx.get(&entry.pc).ok_or_else(|| {
            ExecutionError::RvrExecution(format!("program-log pc {:#x} is unreachable", entry.pc))
        })?;
        frequencies[filtered_idx] =
            frequencies[filtered_idx]
                .checked_add(1)
                .ok_or(ExecutionError::RvrExecution(
                    "program execution frequency overflowed u32".to_string(),
                ))?;
    }
    Ok(frequencies)
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
