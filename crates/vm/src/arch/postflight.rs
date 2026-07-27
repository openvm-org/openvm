use std::{
    collections::{hash_map::Entry, BTreeMap},
    ops::Range,
};

use openvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::Field;
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::{PreflightHistory, BLOCK_FE_WIDTH};

const PREDECESSOR_SEED_BIT: u32 = 1 << 31;
const PREDECESSOR_INDEX_MASK: u32 = !PREDECESSOR_SEED_BIT;

#[derive(Clone, Copy, Debug)]
pub struct PostflightStep {
    pub program_index: u32,
    pub memory_start: u32,
    pub memory_end: u32,
}

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

/// Read-only indexes derived from one serial preflight history.
///
/// Steps are grouped by opcode for parallel trace generation. Memory remains
/// in chronological order; predecessor indexes resolve the value immediately
/// before each timed access without reconstructing mutable RAM.
pub struct Postflight<'a, F> {
    program: &'a Program<F>,
    history: &'a PreflightHistory,
    steps: Vec<PostflightStep>,
    opcode_ranges: BTreeMap<u32, Range<usize>>,
    memory_predecessors: Vec<u32>,
}

#[derive(Debug, Error)]
#[error("invalid preflight history: {0}")]
pub struct PostflightError(String);

impl PostflightError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl<'a, F: Field> Postflight<'a, F> {
    pub fn new(
        program: &'a Program<F>,
        history: &'a PreflightHistory,
        exit_code: Option<u32>,
    ) -> Result<Self, PostflightError> {
        let memory_starts = memory_starts(history)?;
        let mut opcode_counts = BTreeMap::<u32, usize>::new();
        let mut opcodes = Vec::with_capacity(memory_starts.len() - 1);
        for program_index in 0..memory_starts.len() - 1 {
            let opcode = resolve_instruction(program, history, program_index)?.opcode;
            validate_step(history, program_index, opcode, exit_code)?;
            *opcode_counts.entry(opcode.as_usize() as u32).or_default() += 1;
            opcodes.push(opcode);
        }
        validate_endpoint(program, history, exit_code)?;

        let mut opcode_ranges = BTreeMap::new();
        let mut cursor = 0usize;
        for (&opcode, &count) in &opcode_counts {
            opcode_ranges.insert(opcode, cursor..cursor + count);
            cursor += count;
        }
        let mut next = opcode_ranges
            .iter()
            .map(|(&opcode, range)| (opcode, range.start))
            .collect::<BTreeMap<_, _>>();
        let mut steps = vec![
            PostflightStep {
                program_index: 0,
                memory_start: 0,
                memory_end: 0,
            };
            opcodes.len()
        ];
        for (program_index, opcode) in opcodes.into_iter().enumerate() {
            let opcode = opcode.as_usize() as u32;
            let destination = next
                .get_mut(&opcode)
                .expect("opcode count was collected above");
            steps[*destination] = PostflightStep {
                program_index: u32::try_from(program_index)
                    .map_err(|_| PostflightError::new("program log exceeds u32::MAX steps"))?,
                memory_start: memory_starts[program_index],
                memory_end: memory_starts[program_index + 1],
            };
            *destination += 1;
        }

        Ok(Self {
            program,
            history,
            steps,
            opcode_ranges,
            memory_predecessors: memory_predecessors(history)?,
        })
    }

    pub fn steps(&self, opcode: VmOpcode) -> &[PostflightStep] {
        self.opcode_ranges
            .get(&(opcode.as_usize() as u32))
            .map_or(&[], |range| &self.steps[range.clone()])
    }

    pub fn instruction(&self, step: PostflightStep) -> &Instruction<F> {
        resolve_instruction(self.program, self.history, step.program_index as usize)
            .expect("postflight validated every program event")
    }

    pub fn pc(&self, step: PostflightStep) -> u32 {
        self.history.program[step.program_index as usize].pc
    }

    pub fn timestamp(&self, step: PostflightStep) -> u32 {
        self.history.program[step.program_index as usize].timestamp
    }

    pub fn memory(&self, step: PostflightStep) -> &[rvr_state::PreflightMemoryEvent] {
        &self.history.memory.accesses[step.memory_start as usize..step.memory_end as usize]
    }

    pub fn replay(&self, step: PostflightStep) -> PostflightReplay<'_, 'a, F> {
        PostflightReplay {
            postflight: self,
            step,
            memory_cursor: step.memory_start as usize,
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
}

impl<F: Field> PostflightReplay<'_, '_, F> {
    pub fn read_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> Result<U16Access, PostflightError> {
        self.access_u16(address_space, pointer, None)
    }

    pub fn write_u16(
        &mut self,
        address_space: u32,
        pointer: u32,
        expected_value: [u16; BLOCK_FE_WIDTH],
    ) -> Result<U16Access, PostflightError> {
        self.access_u16(address_space, pointer, Some(expected_value))
    }

    pub fn advance_timestamp(&mut self, slots: u32) -> Result<(), PostflightError> {
        self.timestamp = self
            .timestamp
            .checked_add(slots)
            .ok_or_else(|| PostflightError::new("logical timestamp overflow"))?;
        Ok(())
    }

    pub fn finish(self, expected_next_pc: u32) -> Result<(), PostflightError> {
        let program_index = self.step.program_index as usize;
        let actual_next = self.postflight.history.program[program_index + 1];
        if self.memory_cursor != self.step.memory_end as usize {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} left {} memory events unread",
                self.postflight.pc(self.step),
                self.step.memory_end as usize - self.memory_cursor
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
        expected_write: Option<[u16; BLOCK_FE_WIDTH]>,
    ) -> Result<U16Access, PostflightError> {
        if self.memory_cursor >= self.step.memory_end as usize {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} has too few memory events",
                self.postflight.pc(self.step)
            )));
        }
        let event = self.postflight.history.memory.accesses[self.memory_cursor];
        let is_write = expected_write.is_some();
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
        if expected_write.is_some_and(|expected| expected != event.value) {
            return Err(PostflightError::new(format!(
                "instruction at PC {:#x} logged an unexpected write at timestamp {}",
                self.postflight.pc(self.step),
                self.timestamp
            )));
        }
        let access = U16Access {
            value: event.value,
            previous_value: self.postflight.previous_u16(self.memory_cursor),
            previous_timestamp: self.postflight.previous_timestamp(self.memory_cursor),
            timestamp: self.timestamp,
        };
        self.memory_cursor += 1;
        self.timestamp = self
            .timestamp
            .checked_add(1)
            .ok_or_else(|| PostflightError::new("logical timestamp overflow"))?;
        Ok(access)
    }
}

fn memory_starts(history: &PreflightHistory) -> Result<Vec<u32>, PostflightError> {
    if history.program.is_empty() {
        return Err(PostflightError::new(
            "program log must contain a final sentinel",
        ));
    }
    if history.program[0].timestamp != 1 {
        return Err(PostflightError::new(
            "segment program log must start at timestamp 1",
        ));
    }

    let mut memory_cursor = 0usize;
    let mut previous_memory_timestamp = None;
    let mut starts = Vec::with_capacity(history.program.len());
    for (program_index, boundary) in history.program.windows(2).enumerate() {
        let [from, to] = boundary else { unreachable!() };
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
    let pc = history.program[program_index].pc;
    let delta = pc.checked_sub(program.pc_base).ok_or_else(|| {
        PostflightError::new(format!("program log PC {pc:#x} precedes the program base"))
    })?;
    if delta % DEFAULT_PC_STEP != 0 {
        return Err(PostflightError::new(format!(
            "program log PC {pc:#x} is not instruction-aligned"
        )));
    }
    program
        .get_instruction_and_debug_info((delta / DEFAULT_PC_STEP) as usize)
        .map(|(instruction, _)| instruction)
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

fn memory_predecessors(history: &PreflightHistory) -> Result<Vec<u32>, PostflightError> {
    let memory = &history.memory.accesses;
    let seeds = &history.memory.initial_writes;
    if memory.len() >= PREDECESSOR_INDEX_MASK as usize
        || seeds.len() >= PREDECESSOR_INDEX_MASK as usize
    {
        return Err(PostflightError::new(
            "memory or initial-write log exceeds packed predecessor indexes",
        ));
    }

    let mut seed_by_block = FxHashMap::default();
    for (index, seed) in seeds.iter().enumerate() {
        let key = (seed.address_space, seed.pointer);
        if seed_by_block
            .insert(key, u32::try_from(index).unwrap())
            .is_some()
        {
            return Err(PostflightError::new(format!(
                "duplicate initial-write seed for AS={} pointer={}",
                seed.address_space, seed.pointer
            )));
        }
    }

    let mut last_event = FxHashMap::<(u32, u32), u32>::default();
    let mut predecessors = Vec::with_capacity(memory.len());
    for (event_index, event) in memory.iter().enumerate() {
        let key = (event.address_space(), event.pointer);
        let event_index = u32::try_from(event_index).unwrap();
        let predecessor = match last_event.entry(key) {
            Entry::Occupied(mut previous) => {
                let predecessor = *previous.get() + 1;
                previous.insert(event_index);
                predecessor
            }
            Entry::Vacant(vacant) => {
                let predecessor = if event.is_write() {
                    let seed_index = seed_by_block.remove(&key).ok_or_else(|| {
                        PostflightError::new(format!(
                            "first event is a write without a seed for AS={} pointer={}",
                            key.0, key.1
                        ))
                    })?;
                    PREDECESSOR_SEED_BIT | seed_index
                } else {
                    0
                };
                vacant.insert(event_index);
                predecessor
            }
        };
        predecessors.push(predecessor);
    }
    if !seed_by_block.is_empty() {
        return Err(PostflightError::new(format!(
            "{} initial-write seeds are not referenced",
            seed_by_block.len()
        )));
    }
    Ok(predecessors)
}
