use std::{
    collections::{hash_map::Entry, BTreeMap},
    ops::Range,
};

use openvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};
use rustc_hash::FxHashMap;
use thiserror::Error;
use tracing::instrument;

use super::{
    ExecutionState, MemoryCellType, MemoryConfig, PreflightFieldBlock, PreflightHistory,
    PreflightMemoryEvent, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH,
};
use crate::system::{TouchedBlock, TouchedMemory};

#[cfg(any(test, feature = "test-utils"))]
mod testing;

pub(crate) const PREDECESSOR_SEED_BIT: u32 = 1 << 31;
const PREDECESSOR_INDEX_MASK: u32 = !PREDECESSOR_SEED_BIT;

#[inline]
pub(crate) const fn memory_key(address_space: u32, pointer: u32) -> u64 {
    ((address_space as u64) << 32) | pointer as u64
}

#[derive(Clone, Copy, Debug)]
pub struct PostflightStep(u32);

pub struct PostflightReplay<'a, 'history, F> {
    postflight: &'a Postflight<'history, F>,
    step: PostflightStep,
    memory_cursor: usize,
    timestamp: u32,
}

pub struct U16Access {
    pub value: [u16; BLOCK_FE_WIDTH],
    pub previous_value: [u16; BLOCK_FE_WIDTH],
    pub previous_timestamp: u32,
    pub timestamp: u32,
}

pub struct Field32Access<F> {
    pub value: [F; BLOCK_FE_WIDTH],
    pub previous_value: [F; BLOCK_FE_WIDTH],
    pub previous_timestamp: u32,
    pub timestamp: u32,
}

/// Fixed program metadata prepared once and reused by CPU postflight across segments.
pub struct PostflightProgramIndex {
    dense_rows: Vec<u32>,
    num_rows: usize,
}

impl PostflightProgramIndex {
    pub(crate) fn new<F>(program: &Program<F>) -> Result<Self, PostflightError> {
        let mut num_rows = 0u32;
        let dense_rows = program
            .instructions_and_debug_infos
            .iter()
            .map(|instruction| {
                if instruction.is_some() {
                    let row = num_rows;
                    num_rows = num_rows.checked_add(1).ok_or_else(|| {
                        PostflightError::new("program contains more than u32::MAX instructions")
                    })?;
                    Ok(row)
                } else {
                    Ok(u32::MAX)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dense_rows,
            num_rows: num_rows as usize,
        })
    }
}

/// Fills one contiguous range of trace rows from independent preflight steps.
///
/// Rows may be filled concurrently. If `fill` returns an error, other rows may
/// already have run and mutated shared lookup state. The caller must discard
/// the trace-generation session after an error.
pub fn fill_trace_rows<F, E>(
    trace: &mut RowMajorMatrix<F>,
    row_start: usize,
    steps: &[PostflightStep],
    fill: impl Fn(&mut [F], PostflightStep) -> Result<(), E> + Sync,
) -> Result<(), E>
where
    F: Send,
    E: Send,
{
    if steps.is_empty() {
        return Ok(());
    }

    let width = trace.width;
    let start = row_start * width;
    let end = start + steps.len() * width;
    trace.values[start..end]
        .par_chunks_exact_mut(width)
        .zip(steps.par_iter().copied())
        .try_for_each(|(row, step)| fill(row, step))
}

/// Read-only indexes derived from one serial preflight history.
///
/// Steps are grouped by opcode for parallel trace generation. Memory remains
/// in chronological order; predecessor indexes resolve the value immediately
/// before each timed access without reconstructing mutable RAM.
pub struct Postflight<'a, F> {
    program: &'a Program<F>,
    history: &'a PreflightHistory,
    memory_config: MemoryConfig,
    exit_code: Option<u32>,
    steps: Vec<PostflightStep>,
    memory_starts: Vec<u32>,
    opcode_ranges: BTreeMap<u32, Range<usize>>,
    filtered_exec_frequencies: Vec<u32>,
    memory_predecessors: Vec<u32>,
    touched_memory: TouchedMemory<F>,
}

#[derive(Debug, Error)]
#[error("invalid preflight history: {0}")]
pub struct PostflightError(String);

impl PostflightError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl<'a, F: PrimeField32> Postflight<'a, F> {
    pub fn new(
        program: &'a Program<F>,
        history: &'a PreflightHistory,
        memory_config: &MemoryConfig,
        exit_code: Option<u32>,
    ) -> Result<Self, PostflightError> {
        let program_index = PostflightProgramIndex::new(program)?;
        Self::new_inner(
            program,
            &program_index,
            history,
            memory_config,
            exit_code,
            true,
        )
    }

    pub(crate) fn new_prepared(
        program: &'a Program<F>,
        program_index: &PostflightProgramIndex,
        history: &'a PreflightHistory,
        memory_config: &MemoryConfig,
        exit_code: Option<u32>,
    ) -> Result<Self, PostflightError> {
        Self::new_inner(
            program,
            program_index,
            history,
            memory_config,
            exit_code,
            true,
        )
    }

    #[instrument(name = "postflight", skip_all)]
    fn new_inner(
        program: &'a Program<F>,
        program_index: &PostflightProgramIndex,
        history: &'a PreflightHistory,
        memory_config: &MemoryConfig,
        exit_code: Option<u32>,
        validate_final_pc: bool,
    ) -> Result<Self, PostflightError> {
        validate_memory_config(memory_config)?;
        if program_index.dense_rows.len() != program.instructions_and_debug_infos.len() {
            return Err(PostflightError::new(
                "postflight program index does not match the program",
            ));
        }
        let memory_starts = memory_starts(history, memory_config)?;
        let mut opcode_counts = FxHashMap::<u32, usize>::default();
        let mut filtered_exec_frequencies = vec![0u32; program_index.num_rows];
        let mut opcodes = Vec::with_capacity(memory_starts.len() - 1);
        for program_event_index in 0..memory_starts.len() - 1 {
            let slot = resolve_program_slot(program, history, program_event_index)?;
            let instruction = &program.instructions_and_debug_infos[slot]
                .as_ref()
                .expect("resolve_program_slot rejects gaps")
                .0;
            let opcode = instruction.opcode;
            validate_step(history, program_event_index, opcode, exit_code)?;
            if opcode == SystemOpcode::TERMINATE.global_opcode()
                && exit_code != Some(instruction.c.as_canonical_u32())
            {
                return Err(PostflightError::new(
                    "TERMINATE exit code does not match the fetched instruction",
                ));
            }
            let opcode = u32::try_from(opcode.as_usize())
                .map_err(|_| PostflightError::new("instruction opcode exceeds u32::MAX"))?;
            *opcode_counts.entry(opcode).or_default() += 1;
            let dense_row = program_index.dense_rows[slot];
            if dense_row == u32::MAX {
                return Err(PostflightError::new(
                    "postflight program index marks a defined instruction as empty",
                ));
            }
            let frequency = &mut filtered_exec_frequencies[dense_row as usize];
            *frequency = frequency
                .checked_add(1)
                .ok_or_else(|| PostflightError::new("program frequency exceeds u32::MAX"))?;
            opcodes.push(opcode);
        }
        if validate_final_pc {
            validate_endpoint(program, history, exit_code)?;
        }

        let mut opcode_ranges = BTreeMap::new();
        let mut opcode_counts = opcode_counts.into_iter().collect::<Vec<_>>();
        opcode_counts.sort_unstable_by_key(|&(opcode, _)| opcode);
        let mut cursor = 0usize;
        let mut next = FxHashMap::with_capacity_and_hasher(opcode_counts.len(), Default::default());
        for (opcode, count) in opcode_counts {
            opcode_ranges.insert(opcode, cursor..cursor + count);
            next.insert(opcode, cursor);
            cursor += count;
        }
        let mut steps = vec![PostflightStep(0); opcodes.len()];
        for (program_index, opcode) in opcodes.into_iter().enumerate() {
            let destination = next
                .get_mut(&opcode)
                .expect("opcode count was collected above");
            steps[*destination] = PostflightStep(
                u32::try_from(program_index)
                    .map_err(|_| PostflightError::new("program log exceeds u32::MAX steps"))?,
            );
            *destination += 1;
        }

        let (memory_predecessors, touched_memory) = memory_index::<F>(history, memory_config)?;
        Ok(Self {
            program,
            history,
            memory_config: memory_config.clone(),
            exit_code,
            steps,
            memory_starts,
            opcode_ranges,
            filtered_exec_frequencies,
            memory_predecessors,
            touched_memory,
        })
    }

    pub fn steps(&self, opcode: VmOpcode) -> &[PostflightStep] {
        self.opcode_ranges
            .get(&(opcode.as_usize() as u32))
            .map_or(&[], |range| &self.steps[range.clone()])
    }

    #[cfg(feature = "metrics")]
    pub(crate) fn executed_opcodes(&self) -> impl Iterator<Item = VmOpcode> + '_ {
        self.opcode_ranges
            .keys()
            .map(|&opcode| VmOpcode::from_usize(opcode as usize))
    }

    #[cfg(feature = "metrics")]
    pub(crate) fn opcode_count(&self, opcode: VmOpcode) -> u64 {
        self.opcode_ranges
            .get(&(opcode.as_usize() as u32))
            .map_or(0, |range| range.len() as u64)
    }

    pub fn instruction(&self, step: PostflightStep) -> &Instruction<F> {
        let pc = self.pc(step);
        debug_assert!(pc >= self.program.pc_base);
        debug_assert!(pc
            .wrapping_sub(self.program.pc_base)
            .is_multiple_of(DEFAULT_PC_STEP));
        let slot = pc.wrapping_sub(self.program.pc_base) as usize / DEFAULT_PC_STEP as usize;
        &self.program.instructions_and_debug_infos[slot]
            .as_ref()
            .expect("postflight validated every program event")
            .0
    }

    pub fn pc(&self, step: PostflightStep) -> u32 {
        self.history.program[step.0 as usize].pc
    }

    pub fn timestamp(&self, step: PostflightStep) -> u32 {
        self.history.program[step.0 as usize].timestamp
    }

    pub fn from_state(&self) -> ExecutionState<u32> {
        let first = self.history.program[0];
        ExecutionState::new(first.pc, first.timestamp)
    }

    pub fn to_state(&self) -> ExecutionState<u32> {
        let last = *self
            .history
            .program
            .last()
            .expect("postflight requires a final sentinel");
        ExecutionState::new(last.pc, last.timestamp)
    }

    pub fn exit_code(&self) -> Option<u32> {
        self.exit_code
    }

    pub fn filtered_exec_frequencies(&self) -> &[u32] {
        &self.filtered_exec_frequencies
    }

    pub fn touched_memory(&self) -> &TouchedMemory<F> {
        &self.touched_memory
    }

    pub fn replay(&self, step: PostflightStep) -> PostflightReplay<'_, 'a, F> {
        let memory_cursor = self.memory_starts[step.0 as usize] as usize;
        PostflightReplay {
            postflight: self,
            step,
            memory_cursor,
            timestamp: self.timestamp(step),
        }
    }

    fn previous_timestamp(&self, event_index: usize) -> u32 {
        let predecessor = self.memory_predecessors[event_index];
        if predecessor == 0 || predecessor & PREDECESSOR_SEED_BIT != 0 {
            0
        } else {
            self.history.memory.accesses[predecessor as usize - 1].timestamp
        }
    }

    fn previous_u16(&self, event_index: usize) -> [u16; BLOCK_FE_WIDTH] {
        let predecessor = self.memory_predecessors[event_index];
        if predecessor == 0 {
            self.history.memory.accesses[event_index].value
        } else if predecessor & PREDECESSOR_SEED_BIT != 0 {
            self.history.memory.initial_writes[(predecessor & PREDECESSOR_INDEX_MASK) as usize]
                .initial_value
        } else {
            self.history.memory.accesses[predecessor as usize - 1].value
        }
    }

    fn field_value(&self, event_index: usize) -> [F; BLOCK_FE_WIDTH] {
        decode_field_block::<F>(
            self.history.memory.field_values
                [field_reference(self.history.memory.accesses[event_index].value)],
        )
    }

    fn previous_field32(&self, event_index: usize) -> [F; BLOCK_FE_WIDTH] {
        let predecessor = self.memory_predecessors[event_index];
        if predecessor == 0 {
            self.field_value(event_index)
        } else if predecessor & PREDECESSOR_SEED_BIT != 0 {
            let seed =
                self.history.memory.initial_writes[(predecessor & PREDECESSOR_INDEX_MASK) as usize];
            decode_field_block::<F>(
                self.history.memory.field_initial_values[field_reference(seed.initial_value)],
            )
        } else {
            self.field_value(predecessor as usize - 1)
        }
    }
}

impl<F: PrimeField32> PostflightReplay<'_, '_, F> {
    pub fn read_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> Result<U16Access, PostflightError> {
        self.access_u16(address_space, pointer, false, None)
    }

    pub fn write_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
        expected_value: [u16; BLOCK_FE_WIDTH],
    ) -> Result<U16Access, PostflightError> {
        self.access_u16(address_space, pointer, true, Some(expected_value))
    }

    /// Consumes a timed write whose value came from execution advice rather
    /// than deterministic instruction semantics.
    pub fn write_observed_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> Result<U16Access, PostflightError> {
        self.access_u16(address_space, pointer, true, None)
    }

    pub fn read_field32(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> Result<Field32Access<F>, PostflightError> {
        self.access_field32(address_space, pointer, false, None)
    }

    pub fn write_field32(
        &mut self,
        address_space: u32,
        pointer: u32,
        expected_value: [F; BLOCK_FE_WIDTH],
    ) -> Result<Field32Access<F>, PostflightError> {
        self.access_field32(address_space, pointer, true, Some(expected_value))
    }

    /// Resolves an untimed peek from the memory version immediately after the
    /// already-consumed timed-event prefix. Peeks append nothing and do not
    /// advance the logical timestamp. A proof-visible peek must be anchored by
    /// a timed event in the same instruction; peek-only advice is not replayed.
    pub fn peek_u16(
        &self,
        address_space: u32,
        pointer: u32,
    ) -> Result<[u16; BLOCK_FE_WIDTH], PostflightError> {
        self.validate_access_layout(address_space, MemoryCellType::U16)?;
        let program_index = self.step.0 as usize;
        let memory_start = self.postflight.memory_starts[program_index] as usize;
        let memory_end = self.postflight.memory_starts[program_index + 1] as usize;
        let matches = |event: &&rvr_state::PreflightMemoryEvent| {
            event.address_space() == address_space && event.pointer == pointer
        };
        if let Some(event) = self.postflight.history.memory.accesses
            [memory_start..self.memory_cursor]
            .iter()
            .rev()
            .find(matches)
        {
            return Ok(event.value);
        }
        if let Some(offset) = self.postflight.history.memory.accesses
            [self.memory_cursor..memory_end]
            .iter()
            .position(|event| event.address_space() == address_space && event.pointer == pointer)
        {
            return Ok(self.postflight.previous_u16(self.memory_cursor + offset));
        }
        Err(PostflightError::new(format!(
            "instruction at PC {:#x} peeked AS={} pointer={} without a timed event",
            self.postflight.pc(self.step),
            address_space,
            pointer
        )))
    }

    pub fn advance_timestamp(&mut self, slots: u32) -> Result<(), PostflightError> {
        self.timestamp = self
            .timestamp
            .checked_add(slots)
            .ok_or_else(|| PostflightError::new("logical timestamp overflow"))?;
        Ok(())
    }

    pub fn finish(self, expected_next_pc: u32) -> Result<(), PostflightError> {
        let program_index = self.step.0 as usize;
        let actual_next = self.postflight.history.program[program_index + 1];
        let memory_end = self.postflight.memory_starts[program_index + 1] as usize;
        if self.memory_cursor != memory_end {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} left {} memory events unread",
                self.postflight.pc(self.step),
                memory_end - self.memory_cursor
            )));
        }
        if self.timestamp != actual_next.timestamp {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} ended at timestamp {}, expected {}",
                self.postflight.pc(self.step),
                self.timestamp,
                actual_next.timestamp
            )));
        }
        if expected_next_pc != actual_next.pc {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} continued to PC {:#x}, expected {expected_next_pc:#x}",
                self.postflight.pc(self.step),
                actual_next.pc
            )));
        }
        Ok(())
    }

    fn access_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        expected_write: Option<[u16; BLOCK_FE_WIDTH]>,
    ) -> Result<U16Access, PostflightError> {
        let (event_index, event) =
            self.consume_event(address_space, pointer, is_write, MemoryCellType::U16)?;
        if expected_write.is_some_and(|expected| expected != event.value) {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} logged an unexpected write at timestamp {}",
                self.postflight.pc(self.step),
                event.timestamp
            )));
        }
        let access = U16Access {
            value: event.value,
            previous_value: self.postflight.previous_u16(event_index),
            previous_timestamp: self.postflight.previous_timestamp(event_index),
            timestamp: event.timestamp,
        };
        Ok(access)
    }

    fn access_field32(
        &mut self,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        expected_write: Option<[F; BLOCK_FE_WIDTH]>,
    ) -> Result<Field32Access<F>, PostflightError> {
        let (event_index, event) =
            self.consume_event(address_space, pointer, is_write, MemoryCellType::field32())?;
        let value = self.postflight.field_value(event_index);
        if expected_write.is_some_and(|expected| expected != value) {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} logged an unexpected write at timestamp {}",
                self.postflight.pc(self.step),
                event.timestamp
            )));
        }
        let access = Field32Access {
            value,
            previous_value: self.postflight.previous_field32(event_index),
            previous_timestamp: self.postflight.previous_timestamp(event_index),
            timestamp: event.timestamp,
        };
        Ok(access)
    }

    fn consume_event(
        &mut self,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        layout: MemoryCellType,
    ) -> Result<(usize, PreflightMemoryEvent), PostflightError> {
        self.validate_access_layout(address_space, layout)?;
        let memory_end = self.postflight.memory_starts[self.step.0 as usize + 1] as usize;
        if self.memory_cursor >= memory_end {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} has too few memory events",
                self.postflight.pc(self.step)
            )));
        }
        let event = self.postflight.history.memory.accesses[self.memory_cursor];
        if event.timestamp != self.timestamp
            || event.address_space() != address_space
            || event.pointer != pointer
            || event.is_write() != is_write
        {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} has an invalid memory event at timestamp {}",
                self.postflight.pc(self.step),
                self.timestamp
            )));
        }
        let event_index = self.memory_cursor;
        self.memory_cursor += 1;
        self.timestamp = self
            .timestamp
            .checked_add(1)
            .ok_or_else(|| PostflightError::new("logical timestamp overflow"))?;
        Ok((event_index, event))
    }

    fn validate_access_layout(
        &self,
        address_space: u32,
        expected: MemoryCellType,
    ) -> Result<(), PostflightError> {
        let actual = self
            .postflight
            .memory_config
            .addr_spaces
            .get(address_space as usize)
            .map(|config| config.layout);
        if actual != Some(expected) {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} replayed AS={address_space} with the wrong cell layout",
                self.postflight.pc(self.step)
            )));
        }
        Ok(())
    }
}

fn memory_starts(
    history: &PreflightHistory,
    config: &MemoryConfig,
) -> Result<Vec<u32>, PostflightError> {
    if history.program.is_empty() {
        return Err(PostflightError::new(
            "program log must contain a final sentinel",
        ));
    }
    let timestamp_limit = 1u64 << config.timestamp_max_bits;
    if history.program[0].timestamp != 1 {
        return Err(PostflightError::new(
            "segment program log must start at timestamp 1",
        ));
    }
    if u64::from(history.program[0].timestamp) >= timestamp_limit {
        return Err(PostflightError::new(
            "program event 0 timestamp exceeds the configured domain",
        ));
    }

    let mut memory_cursor = 0usize;
    let mut previous_memory_timestamp = None;
    let mut starts = Vec::with_capacity(history.program.len());
    for (program_index, boundary) in history.program.windows(2).enumerate() {
        let [from, to] = boundary else { unreachable!() };
        if u64::from(to.timestamp) >= timestamp_limit {
            return Err(PostflightError::new(format!(
                "program event {} timestamp exceeds the configured domain",
                program_index + 1
            )));
        }
        if to.timestamp < from.timestamp {
            return Err(PostflightError::new(format!(
                "program timestamp moved backwards at step {program_index}"
            )));
        }
        if history
            .memory
            .accesses
            .get(memory_cursor)
            .is_some_and(|event| event.timestamp < from.timestamp)
        {
            return Err(PostflightError::new(format!(
                "memory event {memory_cursor} precedes step {program_index}"
            )));
        }
        starts.push(
            u32::try_from(memory_cursor).map_err(|_| {
                PostflightError::new("memory log contains more than u32::MAX events")
            })?,
        );
        while history
            .memory
            .accesses
            .get(memory_cursor)
            .is_some_and(|event| event.timestamp < to.timestamp)
        {
            let timestamp = history.memory.accesses[memory_cursor].timestamp;
            if previous_memory_timestamp.is_some_and(|previous| previous >= timestamp) {
                return Err(PostflightError::new(
                    "memory timestamps are not strictly increasing",
                ));
            }
            previous_memory_timestamp = Some(timestamp);
            memory_cursor += 1;
        }
    }
    starts.push(
        u32::try_from(memory_cursor)
            .map_err(|_| PostflightError::new("memory log exceeds u32::MAX events"))?,
    );
    if memory_cursor != history.memory.accesses.len() {
        return Err(PostflightError::new(format!(
            "{} memory events occur at or after the final sentinel",
            history.memory.accesses.len() - memory_cursor
        )));
    }
    Ok(starts)
}

fn resolve_instruction<'a, F: Field>(
    program: &'a Program<F>,
    history: &PreflightHistory,
    program_index: usize,
) -> Result<&'a Instruction<F>, PostflightError> {
    let slot = resolve_program_slot(program, history, program_index)?;
    Ok(&program.instructions_and_debug_infos[slot]
        .as_ref()
        .expect("resolve_program_slot rejects gaps")
        .0)
}

fn resolve_program_slot<F: Field>(
    program: &Program<F>,
    history: &PreflightHistory,
    program_index: usize,
) -> Result<usize, PostflightError> {
    let pc = history.program[program_index].pc;
    let delta = pc.checked_sub(program.pc_base).ok_or_else(|| {
        PostflightError::new(format!("program log PC {pc:#x} precedes the program base"))
    })?;
    if delta % DEFAULT_PC_STEP != 0 {
        return Err(PostflightError::new(format!(
            "program log PC {pc:#x} is not instruction-aligned"
        )));
    }
    let slot = (delta / DEFAULT_PC_STEP) as usize;
    program
        .instructions_and_debug_infos
        .get(slot)
        .and_then(Option::as_ref)
        .map(|_| slot)
        .ok_or_else(|| {
            PostflightError::new(format!(
                "program log PC {pc:#x} points to an undefined instruction"
            ))
        })
}

fn validate_step(
    history: &PreflightHistory,
    program_index: usize,
    opcode: VmOpcode,
    exit_code: Option<u32>,
) -> Result<(), PostflightError> {
    let from = history.program[program_index];
    let to = history.program[program_index + 1];
    if opcode == SystemOpcode::TERMINATE.global_opcode() {
        if exit_code.is_none() || program_index + 2 != history.program.len() || from != to {
            return Err(PostflightError::new(
                "TERMINATE must be the final fetched instruction and duplicate the sentinel",
            ));
        }
    } else if to.timestamp == from.timestamp {
        return Err(PostflightError::new(format!(
            "non-TERMINATE instruction {program_index} did not advance the timestamp"
        )));
    }
    Ok(())
}

fn validate_endpoint<F: Field>(
    program: &Program<F>,
    history: &PreflightHistory,
    exit_code: Option<u32>,
) -> Result<(), PostflightError> {
    if exit_code.is_some() {
        if history.program.len() < 2
            || resolve_instruction(program, history, history.program.len() - 2)?.opcode
                != SystemOpcode::TERMINATE.global_opcode()
        {
            return Err(PostflightError::new(
                "terminated history does not end with TERMINATE",
            ));
        }
    } else {
        resolve_instruction(program, history, history.program.len() - 1)?;
    }
    Ok(())
}

fn validate_memory_config(config: &MemoryConfig) -> Result<(), PostflightError> {
    if config.pointer_max_bits > u32::BITS as usize
        || config.addr_space_height >= u32::BITS as usize
        || config.timestamp_max_bits >= u32::BITS as usize
    {
        return Err(PostflightError::new(
            "address-space height, pointer width, and timestamp width must fit u32",
        ));
    }
    if config.pointer_max_bits < BLOCK_FE_WIDTH.ilog2() as usize {
        return Err(PostflightError::new(
            "pointer width is smaller than one memory block",
        ));
    }
    let address_space_count = 1usize
        .checked_shl(config.addr_space_height as u32)
        .ok_or_else(|| PostflightError::new("address-space count overflow"))?;
    let expected_address_spaces = (ADDR_SPACE_OFFSET as usize)
        .checked_add(address_space_count)
        .ok_or_else(|| PostflightError::new("address-space count overflow"))?;
    if config.addr_spaces.len() != expected_address_spaces {
        return Err(PostflightError::new(format!(
            "expected {expected_address_spaces} address-space layouts, found {}",
            config.addr_spaces.len()
        )));
    }
    Ok(())
}

fn validate_memory_block(
    address_space: u32,
    pointer: u32,
    config: &MemoryConfig,
) -> Result<MemoryCellType, PostflightError> {
    let address_space_limit = u64::from(ADDR_SPACE_OFFSET) + (1u64 << config.addr_space_height);
    let address_space_config = config
        .addr_spaces
        .get(address_space as usize)
        .filter(|_| {
            address_space >= ADDR_SPACE_OFFSET && u64::from(address_space) < address_space_limit
        })
        .ok_or_else(|| {
            PostflightError::new(format!("address space {address_space} is out of range"))
        })?;
    if !matches!(
        address_space_config.layout,
        MemoryCellType::U16 | MemoryCellType::F { size: 4 }
    ) {
        return Err(PostflightError::new(format!(
            "address space {address_space} must use u16 or field32 cells"
        )));
    }
    let pointer_limit = 1u64 << config.pointer_max_bits;
    let end = u64::from(pointer)
        .checked_add(BLOCK_FE_WIDTH as u64)
        .ok_or_else(|| PostflightError::new("memory block pointer overflow"))?;
    if !pointer.is_multiple_of(BLOCK_FE_WIDTH as u32)
        || u64::from(pointer) >= pointer_limit
        || end > pointer_limit
        || end > address_space_config.num_cells as u64
    {
        return Err(PostflightError::new(format!(
            "memory block AS={address_space} pointer={pointer} is out of range or misaligned"
        )));
    }
    Ok(address_space_config.layout)
}

fn field_reference(value: [u16; BLOCK_FE_WIDTH]) -> usize {
    usize::try_from(u32::from(value[0]) | (u32::from(value[1]) << 16)).unwrap()
}

fn validate_field_reference(
    value: [u16; BLOCK_FE_WIDTH],
    expected: usize,
    sidecar_len: usize,
    kind: &str,
) -> Result<(), PostflightError> {
    let reference = field_reference(value);
    if value[2] != 0 || value[3] != 0 || reference != expected || reference >= sidecar_len {
        return Err(PostflightError::new(format!(
            "field {kind} must use dense ordered sidecar references"
        )));
    }
    Ok(())
}

fn validate_field_block<F: PrimeField32>(
    block: PreflightFieldBlock,
) -> Result<(), PostflightError> {
    if block.values.iter().any(|&value| value >= F::ORDER_U32) {
        return Err(PostflightError::new(
            "field sidecar contains a non-canonical raw field value",
        ));
    }
    Ok(())
}

fn decode_field_block<F: PrimeField32>(block: PreflightFieldBlock) -> [F; BLOCK_FE_WIDTH] {
    debug_assert!(block.values.iter().all(|&value| value < F::ORDER_U32));
    block.values.map(F::from_u32)
}

fn memory_index<F: PrimeField32>(
    history: &PreflightHistory,
    config: &MemoryConfig,
) -> Result<(Vec<u32>, TouchedMemory<F>), PostflightError> {
    #[derive(Clone, Copy)]
    enum BlockState {
        Seed(u32),
        Event { index: u32, dirty: bool },
    }

    let memory = &history.memory.accesses;
    let seeds = &history.memory.initial_writes;
    if memory.len() >= PREDECESSOR_INDEX_MASK as usize
        || seeds.len() >= PREDECESSOR_INDEX_MASK as usize
    {
        return Err(PostflightError::new(
            "memory or initial-write log exceeds packed predecessor indexes",
        ));
    }

    let mut blocks = FxHashMap::with_capacity_and_hasher(seeds.len(), Default::default());
    let mut field_seed_cursor = 0usize;
    for (index, seed) in seeds.iter().enumerate() {
        if seed.address_space & rvr_state::PREFLIGHT_WRITE_BIT != 0 {
            return Err(PostflightError::new(
                "initial-write seed address space contains the write bit",
            ));
        }
        let layout = validate_memory_block(seed.address_space, seed.pointer, config)?;
        match layout {
            MemoryCellType::U16 => {}
            MemoryCellType::F { size: 4 } => {
                validate_field_reference(
                    seed.initial_value,
                    field_seed_cursor,
                    history.memory.field_initial_values.len(),
                    "initial-write seed",
                )?;
                validate_field_block::<F>(history.memory.field_initial_values[field_seed_cursor])?;
                field_seed_cursor += 1;
            }
            _ => unreachable!("validate_memory_block rejects other layouts"),
        }
        let key = memory_key(seed.address_space, seed.pointer);
        if blocks
            .insert(key, BlockState::Seed(u32::try_from(index).unwrap()))
            .is_some()
        {
            return Err(PostflightError::new(format!(
                "duplicate initial-write seed for AS={} pointer={}",
                seed.address_space, seed.pointer
            )));
        }
    }
    if field_seed_cursor != history.memory.field_initial_values.len() {
        return Err(PostflightError::new(
            "field initial-value sidecar contains unreferenced values",
        ));
    }

    let mut unreferenced_seeds = seeds.len();
    let mut predecessors = Vec::with_capacity(memory.len());
    let mut field_event_cursor = 0usize;
    for (event_index, event) in memory.iter().enumerate() {
        let address_space = event.address_space();
        let layout = validate_memory_block(address_space, event.pointer, config)?;
        match layout {
            MemoryCellType::U16 => {}
            MemoryCellType::F { size: 4 } => {
                validate_field_reference(
                    event.value,
                    field_event_cursor,
                    history.memory.field_values.len(),
                    "memory event",
                )?;
                validate_field_block::<F>(history.memory.field_values[field_event_cursor])?;
                field_event_cursor += 1;
            }
            _ => unreachable!("validate_memory_block rejects other layouts"),
        }

        let key = memory_key(address_space, event.pointer);
        let event_index = u32::try_from(event_index).unwrap();
        let predecessor = match blocks.entry(key) {
            Entry::Occupied(mut state) => match *state.get() {
                BlockState::Seed(seed_index) => {
                    state.insert(BlockState::Event {
                        index: event_index,
                        dirty: event.is_write(),
                    });
                    if event.is_write() {
                        unreferenced_seeds -= 1;
                        PREDECESSOR_SEED_BIT | seed_index
                    } else {
                        0
                    }
                }
                BlockState::Event {
                    index: previous_index,
                    dirty,
                } => {
                    state.insert(BlockState::Event {
                        index: event_index,
                        dirty: dirty || event.is_write(),
                    });
                    previous_index + 1
                }
            },
            Entry::Vacant(state) => {
                if event.is_write() {
                    return Err(PostflightError::new(format!(
                        "first event is a write without a seed for AS={} pointer={}",
                        address_space, event.pointer
                    )));
                }
                state.insert(BlockState::Event {
                    index: event_index,
                    dirty: false,
                });
                0
            }
        };
        predecessors.push(predecessor);
    }
    if field_event_cursor != history.memory.field_values.len() {
        return Err(PostflightError::new(
            "field event sidecar contains unreferenced values",
        ));
    }
    if unreferenced_seeds != 0 {
        return Err(PostflightError::new(format!(
            "{} initial-write seeds are not referenced",
            unreferenced_seeds
        )));
    }

    let mut final_blocks = blocks.into_iter().collect::<Vec<_>>();
    final_blocks.sort_unstable_by_key(|&(key, _)| key);
    let touched_memory = final_blocks
        .into_iter()
        .map(|(_, state)| {
            let BlockState::Event {
                index: event_index,
                dirty,
            } = state
            else {
                unreachable!("all initial-write seeds were referenced")
            };
            let event = history.memory.accesses[event_index as usize];
            let values = match config.addr_spaces[event.address_space() as usize].layout {
                MemoryCellType::U16 => event.value.map(F::from_u16),
                MemoryCellType::F { size: 4 } => decode_field_block::<F>(
                    history.memory.field_values[field_reference(event.value)],
                ),
                _ => unreachable!("memory layouts were validated above"),
            };
            TouchedBlock {
                address_space: event.address_space(),
                ptr: event.pointer,
                is_dirty: u32::from(dirty),
                timestamp: event.timestamp,
                values,
            }
        })
        .collect();
    Ok((predecessors, touched_memory))
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{
        program::Program, riscv::RV64_REGISTER_AS, SystemOpcode, DEFERRAL_AS,
    };
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rvr_state::PREFLIGHT_WRITE_BIT;

    use super::*;
    use crate::arch::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightMemoryLog, PreflightProgramEvent,
    };

    #[test]
    fn fill_trace_rows_skips_empty_ranges() {
        let mut trace = RowMajorMatrix::<BabyBear>::new(Vec::new(), 1);
        fill_trace_rows::<_, ()>(&mut trace, usize::MAX, &[], |_, _| unreachable!()).unwrap();
    }

    #[test]
    fn peek_uses_the_already_consumed_timed_event_prefix() {
        let instruction =
            Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
        let program =
            Program::<BabyBear>::new_without_debug_infos(&[instruction.clone(), instruction], 0);
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 3,
                },
            ],
            memory: PreflightMemoryLog {
                accesses: vec![
                    PreflightMemoryEvent {
                        timestamp: 1,
                        address_space_and_kind: 1 | PREFLIGHT_WRITE_BIT,
                        pointer: 0,
                        value: [2, 0, 0, 0],
                    },
                    PreflightMemoryEvent {
                        timestamp: 2,
                        address_space_and_kind: 1 | PREFLIGHT_WRITE_BIT,
                        pointer: 0,
                        value: [3, 0, 0, 0],
                    },
                ],
                initial_writes: vec![PreflightInitialWrite {
                    address_space: 1,
                    pointer: 0,
                    initial_value: [1, 0, 0, 0],
                }],
                ..Default::default()
            },
        };
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
        let mut replay = postflight.replay(step);

        assert_eq!(replay.peek_u16(1, 0).unwrap(), [1, 0, 0, 0]);
        assert_eq!(replay.write_observed_u16(1, 0).unwrap().value, [2, 0, 0, 0]);
        assert_eq!(replay.peek_u16(1, 0).unwrap(), [2, 0, 0, 0]);
        assert_eq!(replay.write_observed_u16(1, 0).unwrap().value, [3, 0, 0, 0]);
        assert_eq!(replay.peek_u16(1, 0).unwrap(), [3, 0, 0, 0]);
        replay.finish(4).unwrap();
    }

    #[test]
    fn peek_before_a_first_read_uses_the_read_value() {
        let instruction =
            Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
        let program =
            Program::<BabyBear>::new_without_debug_infos(&[instruction.clone(), instruction], 0);
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 2,
                },
            ],
            memory: PreflightMemoryLog {
                accesses: vec![PreflightMemoryEvent {
                    timestamp: 1,
                    address_space_and_kind: 1,
                    pointer: 0,
                    value: [7, 0, 0, 0],
                }],
                ..Default::default()
            },
        };
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
        let mut replay = postflight.replay(step);

        assert_eq!(replay.peek_u16(1, 0).unwrap(), [7, 0, 0, 0]);
        assert!(replay.peek_u16(1, 4).is_err());
        assert_eq!(replay.read_u16(1, 0).unwrap().value, [7, 0, 0, 0]);
        assert_eq!(replay.peek_u16(1, 0).unwrap(), [7, 0, 0, 0]);
        replay.finish(4).unwrap();
    }

    fn field_block(values: [u32; BLOCK_FE_WIDTH]) -> PreflightFieldBlock {
        PreflightFieldBlock { values }
    }

    #[test]
    fn field_sidecars_are_canonical_not_field_representations() {
        let canonical = [1, 2, 3, BabyBear::ORDER_U32 - 1];
        let block = field_block(canonical);
        assert_eq!(block.values, canonical);
        assert_eq!(
            decode_field_block::<BabyBear>(block).map(|value| value.as_canonical_u32()),
            canonical
        );
    }

    fn compact_reference(index: u32) -> [u16; BLOCK_FE_WIDTH] {
        [index as u16, (index >> 16) as u16, 0, 0]
    }

    fn mixed_history() -> (Program<BabyBear>, PreflightHistory) {
        let instruction =
            Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
        let program =
            Program::<BabyBear>::new_without_debug_infos(&[instruction.clone(), instruction], 0);
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 5,
                },
            ],
            memory: PreflightMemoryLog {
                accesses: vec![
                    PreflightMemoryEvent {
                        timestamp: 1,
                        address_space_and_kind: RV64_REGISTER_AS,
                        pointer: 0,
                        value: [1, 2, 3, 4],
                    },
                    PreflightMemoryEvent {
                        timestamp: 2,
                        address_space_and_kind: RV64_REGISTER_AS | PREFLIGHT_WRITE_BIT,
                        pointer: 0,
                        value: [5, 6, 7, 8],
                    },
                    PreflightMemoryEvent {
                        timestamp: 3,
                        address_space_and_kind: DEFERRAL_AS | PREFLIGHT_WRITE_BIT,
                        pointer: 0,
                        value: compact_reference(0),
                    },
                    PreflightMemoryEvent {
                        timestamp: 4,
                        address_space_and_kind: DEFERRAL_AS,
                        pointer: 0,
                        value: compact_reference(1),
                    },
                ],
                initial_writes: vec![PreflightInitialWrite {
                    address_space: DEFERRAL_AS,
                    pointer: 0,
                    initial_value: compact_reference(0),
                }],
                field_values: vec![field_block([21, 22, 23, 24]), field_block([31, 32, 33, 34])],
                field_initial_values: vec![field_block([11, 12, 13, 14])],
            },
        };
        (program, history)
    }

    #[test]
    fn rejects_invalid_program_boundaries() {
        let memory_config = MemoryConfig::default();
        let (program, mut history) = mixed_history();
        history.program[0].pc = 2;
        assert!(Postflight::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("instruction-aligned"));

        let terminate =
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]);
        let phantom =
            Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
        let program = Program::<BabyBear>::new_without_debug_infos(&[terminate, phantom], 0);
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 1,
                },
            ],
            ..Default::default()
        };
        assert!(Postflight::new(&program, &history, &memory_config, Some(0))
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate the sentinel"));
    }

    #[test]
    fn derives_boundary_frequencies_and_mixed_touched_memory() {
        let (program, history) = mixed_history();
        let memory_config = MemoryConfig::default();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();

        assert_eq!(postflight.from_state(), ExecutionState::new(0u32, 1u32));
        assert_eq!(postflight.to_state(), ExecutionState::new(4u32, 5u32));
        assert_eq!(postflight.exit_code(), None);
        assert_eq!(postflight.filtered_exec_frequencies(), [1, 0]);
        assert_eq!(
            postflight
                .touched_memory()
                .iter()
                .map(|block| (
                    block.address_space,
                    block.ptr,
                    block.is_dirty,
                    block.timestamp
                ))
                .collect::<Vec<_>>(),
            [(RV64_REGISTER_AS, 0, 1, 2), (DEFERRAL_AS, 0, 1, 4),]
        );
        assert_eq!(
            postflight.touched_memory()[0]
                .values
                .map(|value| value.as_canonical_u32()),
            [5, 6, 7, 8]
        );
        assert_eq!(
            postflight.touched_memory()[1]
                .values
                .map(|value| value.as_canonical_u32()),
            [31, 32, 33, 34]
        );

        let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
        let mut replay = postflight.replay(step);
        let read = replay.read_u16(RV64_REGISTER_AS, 0).unwrap();
        assert_eq!(read.value, [1, 2, 3, 4]);
        assert_eq!(read.previous_value, read.value);
        let write = replay.write_u16(RV64_REGISTER_AS, 0, [5, 6, 7, 8]).unwrap();
        assert_eq!(write.previous_value, [1, 2, 3, 4]);
        let field_write = replay
            .write_field32(DEFERRAL_AS, 0, [21, 22, 23, 24].map(BabyBear::from_u32))
            .unwrap();
        assert_eq!(
            field_write
                .previous_value
                .map(|value| value.as_canonical_u32()),
            [11, 12, 13, 14]
        );
        let field_read = replay.read_field32(DEFERRAL_AS, 0).unwrap();
        assert_eq!(
            field_read
                .previous_value
                .map(|value| value.as_canonical_u32()),
            [21, 22, 23, 24]
        );
        replay.finish(4).unwrap();
    }

    #[test]
    fn retains_a_terminated_boundary_and_frequency() {
        let exit_code = 7;
        let terminate = Instruction::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [0, 0, exit_code as usize, 0, 0],
        );
        let program = Program::<BabyBear>::new_without_debug_infos(&[terminate], 0);
        let boundary = PreflightProgramEvent {
            pc: 0,
            timestamp: 1,
        };
        let history = PreflightHistory {
            program: vec![boundary, boundary],
            ..Default::default()
        };
        let memory_config = MemoryConfig::default();
        let postflight =
            Postflight::new(&program, &history, &memory_config, Some(exit_code)).unwrap();

        assert_eq!(postflight.from_state(), ExecutionState::new(0u32, 1u32));
        assert_eq!(postflight.to_state(), ExecutionState::new(0u32, 1u32));
        assert_eq!(postflight.exit_code(), Some(exit_code));
        assert_eq!(postflight.filtered_exec_frequencies(), [1]);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn derives_opcode_counts_from_validated_history() {
        let phantom =
            Instruction::from_usize(SystemOpcode::PHANTOM.global_opcode(), [0, 0, 0, 0, 0]);
        let program = Program::<BabyBear>::new_without_debug_infos(
            &[phantom.clone(), phantom.clone(), phantom],
            0,
        );
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 2,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 3,
                },
            ],
            ..Default::default()
        };
        let postflight =
            Postflight::new(&program, &history, &MemoryConfig::default(), None).unwrap();

        assert_eq!(
            postflight.executed_opcodes().collect::<Vec<_>>(),
            [SystemOpcode::PHANTOM.global_opcode()]
        );
        assert_eq!(
            postflight.opcode_count(SystemOpcode::PHANTOM.global_opcode()),
            2
        );
    }

    #[test]
    fn rejects_invalid_memory_domains_and_field_sidecars() {
        let memory_config = MemoryConfig::default();

        let (program, mut history) = mixed_history();
        history.memory.accesses[0].pointer = 1;
        assert!(Postflight::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("misaligned"));

        let (program, mut history) = mixed_history();
        history.memory.accesses[2].value = compact_reference(1);
        assert!(Postflight::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("dense ordered"));

        let (program, mut history) = mixed_history();
        history.memory.field_values[0].values[0] = BabyBear::ORDER_U32;
        assert!(Postflight::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("non-canonical"));

        let (program, mut history) = mixed_history();
        history.program[1].timestamp = 1 << memory_config.timestamp_max_bits;
        assert!(Postflight::new(&program, &history, &memory_config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("timestamp exceeds"));

        let (program, history) = mixed_history();
        let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
        let step = postflight.steps(SystemOpcode::PHANTOM.global_opcode())[0];
        assert!(postflight
            .replay(step)
            .read_u16(DEFERRAL_AS, 0)
            .err()
            .unwrap()
            .to_string()
            .contains("wrong cell layout"));
    }

    #[test]
    fn rejects_invalid_memory_chronology() {
        let read = |timestamp, pointer| PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer,
            value: [0; BLOCK_FE_WIDTH],
        };
        let write = |timestamp, pointer| PreflightMemoryEvent {
            address_space_and_kind: RV64_REGISTER_AS | PREFLIGHT_WRITE_BIT,
            ..read(timestamp, pointer)
        };
        let seed = PreflightInitialWrite {
            address_space: RV64_REGISTER_AS,
            pointer: 0,
            initial_value: [0; BLOCK_FE_WIDTH],
        };
        let history = |accesses, initial_writes| PreflightHistory {
            memory: PreflightMemoryLog {
                accesses,
                initial_writes,
                ..Default::default()
            },
            ..Default::default()
        };
        let error = |history: &PreflightHistory, config: &MemoryConfig| {
            memory_index::<BabyBear>(history, config)
                .unwrap_err()
                .to_string()
        };
        let config = MemoryConfig::default();

        assert!(error(&history(vec![write(1, 0)], vec![]), &config).contains("without a seed"));
        assert!(
            error(&history(vec![write(1, 0)], vec![seed, seed]), &config).contains("duplicate")
        );
        assert!(error(&history(vec![read(1, 0)], vec![seed]), &config).contains("not referenced"));

        let invalid_seed = PreflightInitialWrite {
            address_space: RV64_REGISTER_AS | PREFLIGHT_WRITE_BIT,
            ..seed
        };
        assert!(
            error(&history(vec![write(1, 0)], vec![invalid_seed]), &config)
                .contains("contains the write bit")
        );

        let invalid_address_space = PreflightMemoryEvent {
            address_space_and_kind: 0,
            ..read(1, 0)
        };
        assert!(
            error(&history(vec![invalid_address_space], vec![]), &config).contains("out of range")
        );

        let (program, mut history) = mixed_history();
        history.memory.accesses[1].timestamp = history.memory.accesses[0].timestamp;
        assert!(Postflight::new(&program, &history, &config, None)
            .err()
            .unwrap()
            .to_string()
            .contains("not strictly increasing"));
    }
}
