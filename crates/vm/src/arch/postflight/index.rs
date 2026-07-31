use std::collections::hash_map::Entry;

use openvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode,
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;

use super::{memory_key, PostflightError, PREDECESSOR_INDEX_MASK, PREDECESSOR_SEED_BIT};
use crate::{
    arch::{
        MemoryCellType, MemoryConfig, PreflightFieldBlock, PreflightHistory, ADDR_SPACE_OFFSET,
        BLOCK_FE_WIDTH,
    },
    system::{TouchedBlock, TouchedMemory},
};

/// Fixed program metadata prepared once and reused by CPU postflight across segments.
pub struct PostflightProgramIndex {
    pub(super) dense_rows: Vec<u32>,
    pub(super) num_rows: usize,
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

pub(super) fn memory_starts(
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

pub(super) fn resolve_program_slot<F: Field>(
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

pub(super) fn validate_step(
    history: &PreflightHistory,
    program_index: usize,
    opcode: openvm_instructions::VmOpcode,
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

pub(super) fn validate_endpoint<F: Field>(
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

pub(crate) fn validate_postflight_memory_config(config: &MemoryConfig) -> Result<(), String> {
    if config.pointer_max_bits > u32::BITS as usize
        || config.addr_space_height >= u32::BITS as usize
        || config.timestamp_max_bits >= u32::BITS as usize
    {
        return Err(
            "address-space height, pointer width, and timestamp width must fit u32".to_string(),
        );
    }
    if config.pointer_max_bits < BLOCK_FE_WIDTH.ilog2() as usize {
        return Err("pointer width is smaller than one memory block".to_string());
    }
    let expected_address_spaces = ADDR_SPACE_OFFSET as usize + (1usize << config.addr_space_height);
    if config.addr_spaces.len() != expected_address_spaces {
        return Err(format!(
            "expected {expected_address_spaces} address-space layouts, found {}",
            config.addr_spaces.len()
        ));
    }
    Ok(())
}

pub(super) fn validate_memory_config(config: &MemoryConfig) -> Result<(), PostflightError> {
    validate_postflight_memory_config(config).map_err(PostflightError::new)
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

pub(super) fn field_reference(value: [u16; BLOCK_FE_WIDTH]) -> usize {
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

pub(super) fn decode_field_block<F: PrimeField32>(
    block: PreflightFieldBlock,
) -> [F; BLOCK_FE_WIDTH] {
    debug_assert!(block.values.iter().all(|&value| value < F::ORDER_U32));
    block.values.map(F::from_u32)
}

pub(super) fn memory_index<F: PrimeField32>(
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
