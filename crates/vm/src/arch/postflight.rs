use std::{collections::BTreeMap, ops::Range};

use openvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};
use rustc_hash::FxHashMap;
use thiserror::Error;
use tracing::instrument;

use super::{
    preflight::decode_u8_block, ExecutionState, MemoryCellType, MemoryConfig, PreflightHistory,
    PreflightMemoryEvent, BLOCK_FE_WIDTH,
};
use crate::system::TouchedMemory;

mod index;

#[cfg(feature = "cuda")]
pub(crate) use index::validate_postflight_memory_config;
pub use index::PostflightProgramIndex;
use index::{
    decode_field_block, field_reference, memory_index, memory_starts, resolve_program_slot,
    validate_endpoint, validate_memory_config, validate_step,
};

#[cfg(any(test, feature = "test-utils"))]
mod testing;

/// Exclusive upper bound for indexes packed into a postflight predecessor.
/// The high bit distinguishes an initial-memory entry from a memory event.
pub const POSTFLIGHT_PREDECESSOR_INDEX_LIMIT: u32 = 1 << (u32::BITS - 1);
pub(crate) const PREDECESSOR_SEED_BIT: u32 = POSTFLIGHT_PREDECESSOR_INDEX_LIMIT;
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

pub struct MemoryAccess<T> {
    pub value: [T; BLOCK_FE_WIDTH],
    pub previous_value: [T; BLOCK_FE_WIDTH],
    pub previous_timestamp: u32,
    pub timestamp: u32,
}

pub type U8Access = MemoryAccess<u8>;
pub type U16Access = MemoryAccess<u16>;
pub type Field32Access<F> = MemoryAccess<F>;

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

    fn u8_value(&self, event_index: usize) -> [u8; BLOCK_FE_WIDTH] {
        decode_u8_block(self.history.memory.accesses[event_index].value)
    }

    fn previous_u8(&self, event_index: usize) -> [u8; BLOCK_FE_WIDTH] {
        decode_u8_block(self.previous_inline(event_index))
    }

    fn previous_u16(&self, event_index: usize) -> [u16; BLOCK_FE_WIDTH] {
        self.previous_inline(event_index)
    }

    fn previous_inline(&self, event_index: usize) -> [u16; BLOCK_FE_WIDTH] {
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
    pub fn read_u8(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> Result<U8Access, PostflightError> {
        self.access_u8(address_space, pointer, false, None)
    }

    pub fn write_u8(
        &mut self,
        address_space: u32,
        pointer: u32,
        expected_value: [u8; BLOCK_FE_WIDTH],
    ) -> Result<U8Access, PostflightError> {
        self.access_u8(address_space, pointer, true, Some(expected_value))
    }

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

    fn access_u8(
        &mut self,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        expected_write: Option<[u8; BLOCK_FE_WIDTH]>,
    ) -> Result<U8Access, PostflightError> {
        let (event_index, event) =
            self.consume_event(address_space, pointer, is_write, MemoryCellType::U8)?;
        let value = self.postflight.u8_value(event_index);
        if expected_write.is_some_and(|expected| expected != value) {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} logged an unexpected write at timestamp {}",
                self.postflight.pc(self.step),
                event.timestamp
            )));
        }
        let access = U8Access {
            value,
            previous_value: self.postflight.previous_u8(event_index),
            previous_timestamp: self.postflight.previous_timestamp(event_index),
            timestamp: event.timestamp,
        };
        Ok(access)
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
        let next_timestamp = self
            .timestamp
            .checked_add(1)
            .ok_or_else(|| PostflightError::new("logical timestamp overflow"))?;
        self.memory_cursor += 1;
        self.timestamp = next_timestamp;
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

#[cfg(test)]
mod tests;
